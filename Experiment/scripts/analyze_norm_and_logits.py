"""Measure CrossAttention norm contribution and final-logit JS divergence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Experiment.src.Embd2Adapter.Type_CrossAttention.hypernet_crossattn_models import (  # noqa: E402
    HyperNetController_CrossAttention,
)
from Experiment.src.Embd2Adapter.Type_CrossAttention.hypernet_crossattn_trainer import (  # noqa: E402
    HyperNetCrossAttentionTrainer,
)
from Experiment.src.Embd2Adapter.Type_MeanEmbeddings.hypernet_meanembds_models import (  # noqa: E402
    HyperNetController_MeanEmbds,
)
from Experiment.src.Embd2Adapter.Type_MeanEmbeddings.hypernet_meanembds_trainer import (  # noqa: E402
    HyperNetMeanEmbdsTrainer,
)
from Experiment.src.utils.config import get_hypernet_info  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "Notebooks/outputs/cache/embeddings_cache_validation.npy",
    )
    parser.add_argument(
        "--mean-checkpoint",
        type=Path,
        default=ROOT / "Notebooks/outputs/mean_embds/hypernet_state_dict.pt",
    )
    parser.add_argument(
        "--cross-checkpoint",
        type=Path,
        default=ROOT / "Notebooks/outputs/cross_attn/hypernet_state_dict.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "Experiment/Result/norm_and_logits_diagnostics.json",
    )
    return parser.parse_args()


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def controller_spec(state: dict[str, torch.Tensor], embd_dim: int, device) -> dict:
    projected = state["up_sampler_hyper_net.weight_generator.weight"].shape[1]
    bottleneck = state["up_sampler_hyper_net.bias_generator.weight"].shape[0]
    input_dim = (
        state["up_sampler_hyper_net.weight_generator.weight"].shape[0] // bottleneck
    )
    return {
        "device": device,
        "embd_dim": embd_dim,
        "projected_embd_dim": projected,
        "input_dim": input_dim,
        "reduction_factor": input_dim // bottleneck,
        "task_hidden_dim": state[
            "vec_hypernet.task_embedding_generator.0.weight"
        ].shape[0],
        "num_layers": state["layer_id_embeddings.weight"].shape[0],
    }


def attention_norms(controller, contexts, query) -> tuple[float, float, float]:
    pool = controller.embds_pool
    projected_query = pool.projector(query.unsqueeze(1))
    projected_contexts = pool.projector(contexts)
    self_output, _ = pool.self_mha(
        projected_contexts, projected_contexts, projected_contexts
    )
    contextualized = pool.norm1(projected_contexts + self_output)
    cross_output, _ = pool.cross_mha(
        projected_query, contextualized, contextualized, need_weights=False
    )
    query_norm = projected_query.float().norm(dim=-1).mean().item()
    cross_norm = cross_output.float().norm(dim=-1).mean().item()
    ratio = cross_norm / max(query_norm, 1e-12)
    return query_norm, cross_norm, ratio


def js_divergence(left_logits: torch.Tensor, right_logits: torch.Tensor) -> torch.Tensor:
    left_log_p = F.log_softmax(left_logits.float(), dim=-1)
    right_log_p = F.log_softmax(right_logits.float(), dim=-1)
    log_m = torch.logaddexp(left_log_p, right_log_p) - math.log(2.0)
    left_kl = (left_log_p.exp() * (left_log_p - log_m)).sum(dim=-1)
    right_kl = (right_log_p.exp() * (right_log_p - log_m)).sum(dim=-1)
    return 0.5 * (left_kl + right_kl)


def wrap_models(device, dtype, embd_dim, mean_path, cross_path):
    info = get_hypernet_info()
    model_id = info["model_id"]
    mean_state = torch.load(mean_path, map_location="cpu", weights_only=True)
    cross_state = torch.load(cross_path, map_location="cpu", weights_only=True)
    mean_controller = HyperNetController_MeanEmbds(
        **controller_spec(mean_state, embd_dim, device)
    ).to(device=device, dtype=dtype)
    cross_controller = HyperNetController_CrossAttention(
        **controller_spec(cross_state, embd_dim, device)
    ).to(device=device, dtype=dtype)
    mean_controller.load_state_dict(mean_state)
    cross_controller.load_state_dict(cross_state)

    mean_base = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, local_files_only=True
    ).to(device)
    cross_base = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, local_files_only=True
    ).to(device)
    mean_owner = HyperNetMeanEmbdsTrainer.__new__(HyperNetMeanEmbdsTrainer)
    mean_owner.hypernet = mean_controller
    mean_model = HyperNetMeanEmbdsTrainer.wrap_model(mean_owner, mean_base)
    cross_owner = HyperNetCrossAttentionTrainer.__new__(HyperNetCrossAttentionTrainer)
    cross_owner.hypernet = cross_controller
    cross_owner.info = info
    cross_model = HyperNetCrossAttentionTrainer.wrap_model(cross_owner, cross_base)
    mean_model.eval()
    cross_model.eval()
    return mean_model, cross_model, cross_controller, model_id


def main() -> None:
    args = arguments()
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    context_cache = np.load(args.cache, allow_pickle=True).item()
    query_cache = np.load(
        args.cache.with_name(f"{args.cache.stem}_queries{args.cache.suffix}"),
        allow_pickle=True,
    ).item()
    contexts = context_cache["embeddings"]
    queries = query_cache["embeddings"]
    if len(contexts) != len(queries):
        raise ValueError("Validation context/query embedding caches are not aligned.")

    rng = np.random.default_rng(args.seed)
    count = min(args.samples, len(contexts))
    indices = sorted(rng.choice(len(contexts), count, replace=False).tolist())
    mean_model, cross_model, cross_controller, model_id = wrap_models(
        device,
        dtype,
        int(context_cache["embedding_dim"]),
        args.mean_checkpoint,
        args.cross_checkpoint,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    probe_encoding = tokenizer.apply_chat_template(
        [{
            "role": "user",
            "content": "Answer the question using the supplied context.",
        }],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
    probe = probe_encoding["input_ids"]
    attention_mask = torch.ones_like(probe)

    query_norms = []
    cross_norms = []
    norm_ratios = []
    token_js = []
    sample_js = []
    final_prompt_js = []
    with torch.inference_mode():
        for ordinal, index in enumerate(indices, start=1):
            context_embds = torch.as_tensor(
                np.asarray(contexts[index]), device=device, dtype=dtype
            ).unsqueeze(0)
            query_embd = torch.as_tensor(
                np.asarray(queries[index]), device=device, dtype=dtype
            ).unsqueeze(0)
            q_norm, c_norm, ratio = attention_norms(
                cross_controller, context_embds, query_embd
            )
            query_norms.append(q_norm)
            cross_norms.append(c_norm)
            norm_ratios.append(ratio)

            mean_logits = mean_model(
                input_ids=probe,
                attention_mask=attention_mask,
                embedding=context_embds,
                use_cache=False,
            ).logits
            cross_logits = cross_model(
                input_ids=probe,
                attention_mask=attention_mask,
                context_embds=context_embds,
                query_embd=query_embd,
                use_cache=False,
            ).logits
            probe_js = js_divergence(mean_logits, cross_logits).flatten()
            token_js.extend(probe_js.cpu().tolist())
            sample_js.append(probe_js.mean().item())
            final_prompt_js.append(
                js_divergence(mean_logits[:, -1], cross_logits[:, -1]).item()
            )
            print(f"[{ordinal}/{count}] validation index {index}", flush=True)

    result = {
        "sample_count": count,
        "validation_indices": indices,
        "logit_probe": {
            "text": "Answer the question using the supplied context.",
            "token_count": int(probe.shape[1]),
            "purpose": "Fixed identical token input; validation context/query embeddings remain sample-specific.",
        },
        "norm": {
            "projected_query_l2": summary(query_norms),
            "cross_attention_output_l2": summary(cross_norms),
            "cross_to_query_ratio": summary(norm_ratios),
        },
        "final_logits_js_divergence_nats": {
            "all_probe_positions": summary(token_js),
            "per_sample_probe_mean": summary(sample_js),
            "next_token_after_probe": summary(final_prompt_js),
            "token_count": len(token_js),
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
