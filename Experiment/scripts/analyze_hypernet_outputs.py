"""Compare MeanEmbds and CrossAttention HyperNet conditioning outputs.

This intentionally runs without loading the base causal LM.  It diagnoses the
conditioning path (attention pooling -> generated adapter parameters), which is
both much cheaper and more useful for locating representation collapse.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "Experiment"))

from src.Embd2Adapter.Type_CrossAttention.hypernet_crossattn_models import (  # noqa: E402
    HyperNetController_CrossAttention,
)
from src.Embd2Adapter.Type_MeanEmbeddings.hypernet_meanembds_models import (  # noqa: E402
    HyperNetController_MeanEmbds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=REPOSITORY_ROOT / "Notebooks/outputs/cache/embeddings_cache_validation.npy",
    )
    parser.add_argument(
        "--mean-checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "Notebooks/outputs/mean_embds/hypernet_state_dict.pt",
    )
    parser.add_argument(
        "--cross-checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "Notebooks/outputs/cross_attn/hypernet_state_dict.pt",
    )
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_embeddings(path: Path) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    context_cache = np.load(path, allow_pickle=True).item()
    query_path = path.with_name(f"{path.stem}_queries{path.suffix}")
    query_cache = np.load(query_path, allow_pickle=True).item()
    contexts = context_cache["embeddings"]
    queries = query_cache["embeddings"]
    if len(contexts) != len(queries):
        raise ValueError("Context and query caches have different row counts.")
    return contexts, queries, int(context_cache["embedding_dim"])


def infer_spec(state: dict[str, torch.Tensor], embd_dim: int) -> dict[str, int]:
    projected_dim = state["up_sampler_hyper_net.weight_generator.weight"].shape[1]
    up_weight = state["up_sampler_hyper_net.weight_generator.weight"]
    up_bias = state["up_sampler_hyper_net.bias_generator.weight"]
    bottleneck = up_bias.shape[0]
    input_dim = up_weight.shape[0] // bottleneck
    return {
        "embd_dim": embd_dim,
        "projected_embd_dim": projected_dim,
        "input_dim": input_dim,
        "reduction_factor": input_dim // bottleneck,
        "task_hidden_dim": state["vec_hypernet.task_embedding_generator.0.weight"].shape[0],
        "num_layers": state["layer_id_embeddings.weight"].shape[0],
    }


def flatten_adapter(outputs) -> torch.Tensor:
    values = []
    for adapter in outputs:
        for sampler in adapter:
            values.extend((sampler.weight.reshape(-1), sampler.bias.reshape(-1)))
    return torch.cat(values).float()


def pair_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    delta = left - right
    return {
        "cosine": F.cosine_similarity(left, right, dim=0).clamp(-1.0, 1.0).item(),
        "relative_l2": (delta.norm() / right.norm().clamp_min(1e-12)).item(),
        "rmse": delta.square().mean().sqrt().item(),
    }


def uniform_attention_pool(model, contexts: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Run the trained pooling path with only cross-attention weights ablated."""
    pool = model.embds_pool
    projected_query = pool.projector(query.unsqueeze(1))
    projected_contexts = pool.projector(contexts)
    self_output, _ = pool.self_mha(
        projected_contexts, projected_contexts, projected_contexts
    )
    contextualized = pool.norm1(projected_contexts + self_output)
    _, _, value_weight = pool.cross_mha.in_proj_weight.chunk(3, dim=0)
    if pool.cross_mha.in_proj_bias is None:
        value_bias = None
    else:
        _, _, value_bias = pool.cross_mha.in_proj_bias.chunk(3, dim=0)
    values = F.linear(contextualized, value_weight, value_bias)
    uniform_cross_output = F.linear(
        values.mean(dim=1, keepdim=True),
        pool.cross_mha.out_proj.weight,
        pool.cross_mha.out_proj.bias,
    )
    return pool.norm2(projected_query + uniform_cross_output).squeeze(1)


def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows])),
            "min": float(np.min([row[key] for row in rows])),
            "max": float(np.max([row[key] for row in rows])),
        }
        for key in rows[0]
    }


def main() -> None:
    args = parse_args()
    contexts, queries, embd_dim = load_embeddings(args.cache)
    mean_state = torch.load(args.mean_checkpoint, map_location="cpu", weights_only=True)
    cross_state = torch.load(args.cross_checkpoint, map_location="cpu", weights_only=True)
    mean_spec = infer_spec(mean_state, embd_dim)
    cross_spec = infer_spec(cross_state, embd_dim)
    mean_model = HyperNetController_MeanEmbds(torch.device("cpu"), **mean_spec)
    cross_model = HyperNetController_CrossAttention(torch.device("cpu"), **cross_spec)
    mean_model.load_state_dict(mean_state)
    cross_model.load_state_dict(cross_state)
    mean_model.eval()
    cross_model.eval()

    rng = np.random.default_rng(args.seed)
    count = min(args.samples, len(contexts))
    indices = rng.choice(len(contexts), size=count, replace=False)
    layer_ids = sorted({0, mean_spec["num_layers"] // 2, mean_spec["num_layers"] - 1})
    pooling_rows = []
    adapter_isolation_rows = []
    model_comparison_rows = []

    with torch.inference_mode():
        for index in indices:
            context = torch.as_tensor(np.asarray(contexts[index]), dtype=torch.float32).unsqueeze(0)
            query = torch.as_tensor(np.asarray(queries[index]), dtype=torch.float32).unsqueeze(0)
            mean_embedding = context.mean(dim=1)
            pooled, weights = cross_model.pool_embeddings(
                context, query, return_attention_weights=True
            )
            uniform_pooled = uniform_attention_pool(cross_model, context, query)
            head_weights = weights.squeeze(0).squeeze(1)
            average_weights = head_weights.mean(dim=0)
            normalized_entropy = -(
                average_weights.clamp_min(1e-12) * average_weights.clamp_min(1e-12).log()
            ).sum() / np.log(average_weights.numel()) if average_weights.numel() > 1 else torch.tensor(0.0)
            pooling_rows.append({
                **pair_metrics(pooled, uniform_pooled),
                "attention_max": average_weights.max().item(),
                "attention_normalized_entropy": normalized_entropy.item(),
                "attention_head_std": head_weights.std(dim=0).mean().item(),
            })
            for layer_id in layer_ids:
                cross_attended = flatten_adapter(
                    cross_model(pooled_embedding=pooled, layer_id=layer_id)
                )
                cross_mean_ablation = flatten_adapter(
                    cross_model(pooled_embedding=uniform_pooled, layer_id=layer_id)
                )
                mean_output = flatten_adapter(mean_model(mean_embedding, layer_id))
                adapter_isolation_rows.append(pair_metrics(cross_attended, cross_mean_ablation))
                model_comparison_rows.append(pair_metrics(cross_attended, mean_output))

    result = {
        "sample_count": count,
        "layer_ids": layer_ids,
        "interpretation": {
            "pooling_vs_uniform_attention": "Does learned cross-attention create a distinct conditioning vector versus uniform cross-attention, with every other operation fixed?",
            "cross_attention_vs_uniform_ablation": "Within the same CrossAttention HyperNet, does that attention-only distinction survive into generated adapter parameters?",
            "cross_model_vs_mean_model": "End-to-end adapter-parameter difference; also includes independently trained model weights.",
        },
        "pooling_vs_uniform_attention": summarize(pooling_rows),
        "cross_attention_vs_uniform_ablation": summarize(adapter_isolation_rows),
        "cross_model_vs_mean_model": summarize(model_comparison_rows),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
