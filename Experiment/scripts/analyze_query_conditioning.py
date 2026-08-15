"""Inspect query diversity and QK discrimination in a CrossAttention checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Experiment"))

from src.Embd2Adapter.Type_CrossAttention.hypernet_crossattn_models import (  # noqa: E402
    HyperNetController_CrossAttention,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument(
        "--cache", type=Path,
        default=ROOT / "Notebooks/outputs/cache/embeddings_cache_validation.npy",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "Notebooks/outputs/cross_attn/checkpoint-1650/model.safetensors",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "Experiment/Result/query_checkpoint_1650_diagnostics.json",
    )
    return parser.parse_args()


def load_state(path):
    with safe_open(path, framework="pt", device="cpu") as file:
        return {
            key.removeprefix("hypernet."): file.get_tensor(key)
            for key in file.keys() if key.startswith("hypernet.")
        }


def infer_spec(state, embd_dim):
    projected = state["up_sampler_hyper_net.weight_generator.weight"].shape[1]
    bottleneck = state["up_sampler_hyper_net.bias_generator.weight"].shape[0]
    input_dim = state["up_sampler_hyper_net.weight_generator.weight"].shape[0] // bottleneck
    return dict(
        device=torch.device("cpu"), embd_dim=embd_dim,
        projected_embd_dim=projected, input_dim=input_dim,
        reduction_factor=input_dim // bottleneck,
        task_hidden_dim=state["vec_hypernet.task_embedding_generator.0.weight"].shape[0],
        num_layers=state["layer_id_embeddings.weight"].shape[0],
    )


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(mean=float(a.mean()), std=float(a.std()), min=float(a.min()),
                median=float(np.median(a)), max=float(a.max()))


def cross_sample_cosine(vectors):
    x = F.normalize(torch.cat(vectors, dim=0).float(), dim=-1)
    matrix = x @ x.T
    mask = ~torch.eye(len(x), dtype=torch.bool)
    return matrix[mask].tolist()


def main():
    options = parse_args()
    context_cache = np.load(options.cache, allow_pickle=True).item()
    query_cache = np.load(
        options.cache.with_name(f"{options.cache.stem}_queries{options.cache.suffix}"),
        allow_pickle=True,
    ).item()
    state = load_state(options.checkpoint)
    model = HyperNetController_CrossAttention(
        **infer_spec(state, int(context_cache["embedding_dim"]))
    )
    model.load_state_dict(state)
    model.eval()
    pool, mha = model.embds_pool, model.embds_pool.cross_mha
    rng = np.random.default_rng(options.seed)
    indices = rng.choice(len(context_cache["embeddings"]),
                         min(options.samples, len(context_cache["embeddings"])),
                         replace=False)
    raw_queries, projected_queries, internal_queries = [], [], []
    internal_queries_without_bias = []
    metrics = {key: [] for key in (
        "raw_query_l2", "projected_query_l2", "internal_q_l2_per_head",
        "internal_k_l2_per_head", "qk_score_std_across_chunks",
        "qk_score_range_across_chunks", "cross_output_l2",
        "cross_to_projected_query_norm_ratio", "attention_normalized_entropy",
        "attention_max",
        "q_bias_to_bias_free_signal_norm_ratio",
        "shuffled_query_attention_l1", "shuffled_query_attention_js",
        "shuffled_query_top1_changed",
    )}
    q_weight, k_weight, _ = mha.in_proj_weight.chunk(3, dim=0)
    q_bias, k_bias, _ = mha.in_proj_bias.chunk(3, dim=0)
    head_dim = model.embd_dim // model.embds_pool.num_heads

    with torch.inference_mode():
        for sample_position, index in enumerate(indices):
            context = torch.as_tensor(
                np.asarray(context_cache["embeddings"][index]), dtype=torch.float32
            ).unsqueeze(0)
            query = torch.as_tensor(
                np.asarray(query_cache["embeddings"][index]), dtype=torch.float32
            ).unsqueeze(0)
            projected_q = pool.projector(query.unsqueeze(1))
            projected_c = pool.projector(context)
            self_output, _ = pool.self_mha(projected_c, projected_c, projected_c)
            contextualized = pool.norm1(projected_c + self_output)
            internal_q = F.linear(projected_q, q_weight, q_bias)
            internal_q_without_bias = F.linear(projected_q, q_weight, None)
            internal_k = F.linear(contextualized, k_weight, k_bias)
            q_heads = internal_q.view(1, 1, mha.num_heads, head_dim).transpose(1, 2)
            k_heads = internal_k.view(1, -1, mha.num_heads, head_dim).transpose(1, 2)
            scores = (q_heads @ k_heads.transpose(-1, -2) / math.sqrt(head_dim)).squeeze(0).squeeze(1)
            cross_output, weights = mha(
                projected_q, contextualized, contextualized,
                need_weights=True, average_attn_weights=False,
            )
            average_weights = weights.mean(dim=1).squeeze(0).squeeze(0).float()
            shuffled_index = indices[(sample_position + 1) % len(indices)]
            shuffled_query = torch.as_tensor(
                np.asarray(query_cache["embeddings"][shuffled_index]), dtype=torch.float32
            ).unsqueeze(0)
            shuffled_q = pool.projector(shuffled_query.unsqueeze(1))
            _, shuffled_weights = mha(
                shuffled_q, contextualized, contextualized,
                need_weights=True, average_attn_weights=False,
            )
            shuffled_average = shuffled_weights.mean(dim=1).squeeze(0).squeeze(0).float()
            mixture = 0.5 * (average_weights + shuffled_average)
            entropy = -(average_weights.clamp_min(1e-12)
                        * average_weights.clamp_min(1e-12).log()).sum()
            entropy /= math.log(average_weights.numel())
            raw_queries.append(query)
            projected_queries.append(projected_q.squeeze(1))
            internal_queries.append(internal_q.squeeze(1))
            internal_queries_without_bias.append(internal_q_without_bias.squeeze(1))
            metrics["raw_query_l2"].append(query.norm().item())
            metrics["projected_query_l2"].append(projected_q.norm().item())
            metrics["internal_q_l2_per_head"].extend(
                q_heads.norm(dim=-1).flatten().tolist()
            )
            metrics["internal_k_l2_per_head"].extend(
                k_heads.norm(dim=-1).flatten().tolist()
            )
            metrics["qk_score_std_across_chunks"].extend(
                scores.std(dim=-1).tolist()
            )
            metrics["qk_score_range_across_chunks"].extend(
                (scores.max(dim=-1).values - scores.min(dim=-1).values).tolist()
            )
            metrics["cross_output_l2"].append(cross_output.norm().item())
            metrics["cross_to_projected_query_norm_ratio"].append(
                (cross_output.norm() / projected_q.norm().clamp_min(1e-12)).item()
            )
            metrics["attention_normalized_entropy"].append(entropy.item())
            metrics["attention_max"].append(average_weights.max().item())
            metrics["q_bias_to_bias_free_signal_norm_ratio"].append(
                (q_bias.norm() / internal_q_without_bias.norm().clamp_min(1e-12)).item()
            )
            metrics["shuffled_query_attention_l1"].append(
                (average_weights - shuffled_average).abs().sum().item()
            )
            metrics["shuffled_query_attention_js"].append(
                (0.5 * (
                    (average_weights * (average_weights.clamp_min(1e-12).log()
                     - mixture.clamp_min(1e-12).log())).sum()
                    + (shuffled_average * (shuffled_average.clamp_min(1e-12).log()
                       - mixture.clamp_min(1e-12).log())).sum()
                )).item()
            )
            metrics["shuffled_query_top1_changed"].append(
                float(average_weights.argmax() != shuffled_average.argmax())
            )

    result = {
        "sample_count": len(indices),
        "metrics": {key: summary(value) for key, value in metrics.items()},
        "cross_sample_pairwise_cosine": {
            "raw_query": summary(cross_sample_cosine(raw_queries)),
            "projected_query": summary(cross_sample_cosine(projected_queries)),
            "internal_q": summary(cross_sample_cosine(internal_queries)),
            "internal_q_without_bias": summary(
                cross_sample_cosine(internal_queries_without_bias)
            ),
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    options.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
