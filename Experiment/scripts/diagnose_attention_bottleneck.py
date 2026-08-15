"""Locate collapse in embedding, attention pooling, or HyperNet sensitivity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Experiment"))

from src.Embd2Adapter.Type_CrossAttention.hypernet_crossattn_models import (  # noqa: E402
    HyperNetController_CrossAttention,
)


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "Notebooks/outputs/cache/embeddings_cache_validation.npy",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "Notebooks/outputs/cross_attn/hypernet_state_dict.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "Experiment/Result/attention_bottleneck_diagnostics.json",
    )
    return parser.parse_args()


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


def off_diagonal_cosine(vectors):
    vectors = F.normalize(vectors.float(), dim=-1)
    matrix = vectors @ vectors.transpose(-1, -2)
    n = matrix.shape[-1]
    mask = ~torch.eye(n, dtype=torch.bool)
    return matrix[0][mask].tolist()


def relative_delta(left, right):
    return ((left.float() - right.float()).norm() / right.float().norm().clamp_min(1e-12)).item()


def flatten_adapter(outputs):
    tensors = []
    for adapter in outputs:
        for sampler in adapter:
            tensors.extend((sampler.weight.reshape(-1), sampler.bias.reshape(-1)))
    return torch.cat(tensors).float()


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()), "std": float(array.std()),
        "min": float(array.min()), "median": float(np.median(array)),
        "max": float(array.max()),
    }


def main():
    options = args()
    cache = np.load(options.cache, allow_pickle=True).item()
    query_cache = np.load(
        options.cache.with_name(f"{options.cache.stem}_queries{options.cache.suffix}"),
        allow_pickle=True,
    ).item()
    contexts, queries = cache["embeddings"], query_cache["embeddings"]
    state = torch.load(options.checkpoint, map_location="cpu", weights_only=True)
    model = HyperNetController_CrossAttention(
        **infer_spec(state, int(cache["embedding_dim"]))
    )
    model.load_state_dict(state)
    model.eval()
    pool = model.embds_pool
    rng = np.random.default_rng(options.seed)
    count = min(options.samples, len(contexts))
    indices = rng.choice(len(contexts), size=count, replace=False)
    layers = sorted({0, model.num_layers // 2, model.num_layers - 1})

    metrics = {name: [] for name in (
        "raw_chunk_pairwise_cosine", "projected_chunk_pairwise_cosine",
        "projector_hidden_pairwise_cosine", "projected_without_final_bias_pairwise_cosine",
        "projected_chunk_spread_to_centroid_ratio", "projector_final_bias_to_signal_norm_ratio",
        "contextualized_chunk_pairwise_cosine", "value_pairwise_cosine",
        "attention_normalized_entropy", "attention_max", "attention_top_gap",
        "attention_max_to_min_ratio", "attended_vs_uniform_cross_output_relative_l2",
        "attended_vs_uniform_pool_relative_l2", "onehot_vs_uniform_pool_relative_l2",
        "actual_adapter_relative_delta", "onehot_adapter_relative_delta",
        "actual_hypernet_transmission_ratio", "onehot_hypernet_transmission_ratio",
    )}

    with torch.inference_mode():
        for index in indices:
            context = torch.as_tensor(np.asarray(contexts[index]), dtype=torch.float32).unsqueeze(0)
            query = torch.as_tensor(np.asarray(queries[index]), dtype=torch.float32).unsqueeze(0)
            projected_q = pool.projector(query.unsqueeze(1))
            projected_c = pool.projector(context)
            projector_hidden = pool.projector[1](pool.projector[0](context))
            projected_without_final_bias = F.linear(
                projector_hidden, pool.projector[2].weight, bias=None
            )
            self_out, _ = pool.self_mha(projected_c, projected_c, projected_c)
            contextualized = pool.norm1(projected_c + self_out)
            _, _, value_weight = pool.cross_mha.in_proj_weight.chunk(3, dim=0)
            _, _, value_bias = pool.cross_mha.in_proj_bias.chunk(3, dim=0)
            values = F.linear(contextualized, value_weight, value_bias)
            actual_cross, weights = pool.cross_mha(
                projected_q, contextualized, contextualized,
                need_weights=True, average_attn_weights=False,
            )
            attended_pool = pool.norm2(projected_q + actual_cross).squeeze(1)
            average_weights = weights.mean(dim=1).squeeze(1).squeeze(0).float()
            uniform_value = values.mean(dim=1, keepdim=True)
            top_index = int(average_weights.argmax())
            onehot_value = values[:, top_index:top_index + 1]
            uniform_cross = F.linear(uniform_value, pool.cross_mha.out_proj.weight, pool.cross_mha.out_proj.bias)
            onehot_cross = F.linear(onehot_value, pool.cross_mha.out_proj.weight, pool.cross_mha.out_proj.bias)
            uniform_pool = pool.norm2(projected_q + uniform_cross).squeeze(1)
            onehot_pool = pool.norm2(projected_q + onehot_cross).squeeze(1)

            metrics["raw_chunk_pairwise_cosine"].extend(off_diagonal_cosine(context))
            metrics["projector_hidden_pairwise_cosine"].extend(
                off_diagonal_cosine(projector_hidden)
            )
            metrics["projected_chunk_pairwise_cosine"].extend(off_diagonal_cosine(projected_c))
            metrics["projected_without_final_bias_pairwise_cosine"].extend(
                off_diagonal_cosine(projected_without_final_bias)
            )
            metrics["projector_final_bias_to_signal_norm_ratio"].append(
                (pool.projector[2].bias.float().norm()
                 / projected_without_final_bias.float().norm(dim=-1).mean().clamp_min(1e-12)).item()
            )
            centroid = projected_c.mean(dim=1, keepdim=True)
            metrics["projected_chunk_spread_to_centroid_ratio"].append(
                ((projected_c - centroid).float().norm(dim=-1).mean()
                 / centroid.float().norm(dim=-1).mean().clamp_min(1e-12)).item()
            )
            metrics["contextualized_chunk_pairwise_cosine"].extend(off_diagonal_cosine(contextualized))
            metrics["value_pairwise_cosine"].extend(off_diagonal_cosine(values))
            entropy = -(average_weights.clamp_min(1e-12) * average_weights.clamp_min(1e-12).log()).sum()
            entropy /= np.log(average_weights.numel()) if average_weights.numel() > 1 else 1.0
            sorted_weights = average_weights.sort(descending=True).values
            metrics["attention_normalized_entropy"].append(entropy.item())
            metrics["attention_max"].append(sorted_weights[0].item())
            metrics["attention_top_gap"].append((sorted_weights[0] - sorted_weights[1]).item())
            metrics["attention_max_to_min_ratio"].append(
                (sorted_weights[0] / sorted_weights[-1].clamp_min(1e-12)).item()
            )
            metrics["attended_vs_uniform_cross_output_relative_l2"].append(
                relative_delta(actual_cross, uniform_cross)
            )
            actual_pool_delta = relative_delta(attended_pool, uniform_pool)
            onehot_pool_delta = relative_delta(onehot_pool, uniform_pool)
            metrics["attended_vs_uniform_pool_relative_l2"].append(actual_pool_delta)
            metrics["onehot_vs_uniform_pool_relative_l2"].append(onehot_pool_delta)

            for layer in layers:
                uniform_adapter = flatten_adapter(model(pooled_embedding=uniform_pool, layer_id=layer))
                actual_adapter = flatten_adapter(model(pooled_embedding=attended_pool, layer_id=layer))
                onehot_adapter = flatten_adapter(model(pooled_embedding=onehot_pool, layer_id=layer))
                actual_adapter_delta = relative_delta(actual_adapter, uniform_adapter)
                onehot_adapter_delta = relative_delta(onehot_adapter, uniform_adapter)
                metrics["actual_adapter_relative_delta"].append(actual_adapter_delta)
                metrics["onehot_adapter_relative_delta"].append(onehot_adapter_delta)
                metrics["actual_hypernet_transmission_ratio"].append(
                    actual_adapter_delta / max(actual_pool_delta, 1e-12)
                )
                metrics["onehot_hypernet_transmission_ratio"].append(
                    onehot_adapter_delta / max(onehot_pool_delta, 1e-12)
                )

    result = {
        "sample_count": count,
        "layer_ids_for_hypernet_sensitivity": layers,
        "metrics": {key: summarize(value) for key, value in metrics.items()},
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
