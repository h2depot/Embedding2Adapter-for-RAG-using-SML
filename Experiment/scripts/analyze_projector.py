"""Detailed diagnostics and counterfactual ablations for AttentionPooling.projector."""

from __future__ import annotations

import argparse
import json
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
        default=ROOT / "Notebooks/outputs/cross_attn/hypernet_state_dict.pt",
    )
    parser.add_argument(
        "--trainer-checkpoint", type=Path,
        default=ROOT / "Notebooks/outputs/cross_attn/checkpoint-2400/model.safetensors",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "Experiment/Result/projector_diagnostics.json",
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


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(mean=float(a.mean()), std=float(a.std()), min=float(a.min()),
                median=float(np.median(a)), max=float(a.max()))


def pairwise_cosine(vectors):
    vectors = F.normalize(vectors.float(), dim=-1)
    matrix = vectors @ vectors.transpose(-1, -2)
    mask = ~torch.eye(matrix.shape[-1], dtype=torch.bool)
    return matrix[0][mask].tolist()


def matrix_report(weight):
    singular = torch.linalg.svdvals(weight.float())
    probabilities = singular / singular.sum()
    effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    return {
        "frobenius_norm": weight.float().norm().item(),
        "spectral_norm": singular[0].item(),
        "stable_rank": (singular.square().sum() / singular[0].square()).item(),
        "effective_rank": effective_rank.item(),
        "rank_capacity": int(min(weight.shape)),
        "smallest_to_largest_singular_ratio": (singular[-1] / singular[0]).item(),
    }


def project(pool, inputs, variant):
    first_bias = None if variant == "no_all_biases" else pool.projector[0].bias
    hidden = F.gelu(F.linear(inputs, pool.projector[0].weight, first_bias))
    final_bias = pool.projector[2].bias if variant == "baseline" else None
    output = F.linear(hidden, pool.projector[2].weight, final_bias)
    if variant == "no_final_bias_plus_raw_residual":
        output = output + inputs
    return output


def relative_delta(left, right):
    return ((left.float() - right.float()).norm()
            / right.float().norm().clamp_min(1e-12)).item()


def load_hypernet_state(path):
    if path.suffix == ".safetensors":
        with safe_open(path, framework="pt", device="cpu") as file:
            return {
                key.removeprefix("hypernet."): file.get_tensor(key)
                for key in file.keys()
                if key.startswith("hypernet.")
            }
    return torch.load(path, map_location="cpu", weights_only=True)


def main():
    options = parse_args()
    cache = np.load(options.cache, allow_pickle=True).item()
    query_cache = np.load(
        options.cache.with_name(f"{options.cache.stem}_queries{options.cache.suffix}"),
        allow_pickle=True,
    ).item()
    contexts, queries = cache["embeddings"], query_cache["embeddings"]
    state = load_hypernet_state(options.checkpoint)
    model = HyperNetController_CrossAttention(
        **infer_spec(state, int(cache["embedding_dim"]))
    )
    model.load_state_dict(state)
    model.eval()
    pool = model.embds_pool
    rng = np.random.default_rng(options.seed)
    count = min(options.samples, len(contexts))
    indices = rng.choice(len(contexts), size=count, replace=False)
    variants = ("baseline", "no_final_bias", "no_all_biases", "no_final_bias_plus_raw_residual")
    collected = {
        variant: {key: [] for key in (
            "projected_pairwise_cosine", "contextualized_pairwise_cosine",
            "value_pairwise_cosine", "attention_normalized_entropy",
            "attention_max", "attended_vs_uniform_pool_relative_l2",
            "projected_spread_to_centroid_ratio",
        )} for variant in variants
    }

    with torch.inference_mode():
        for index in indices:
            context = torch.as_tensor(np.asarray(contexts[index]), dtype=torch.float32).unsqueeze(0)
            query = torch.as_tensor(np.asarray(queries[index]), dtype=torch.float32).unsqueeze(0)
            for variant in variants:
                q = project(pool, query.unsqueeze(1), variant)
                c = project(pool, context, variant)
                self_output, _ = pool.self_mha(c, c, c)
                contextualized = pool.norm1(c + self_output)
                actual_cross, weights = pool.cross_mha(
                    q, contextualized, contextualized,
                    need_weights=True, average_attn_weights=False,
                )
                average_weights = weights.mean(dim=1).squeeze(1).squeeze(0).float()
                _, _, value_weight = pool.cross_mha.in_proj_weight.chunk(3, dim=0)
                _, _, value_bias = pool.cross_mha.in_proj_bias.chunk(3, dim=0)
                values = F.linear(contextualized, value_weight, value_bias)
                uniform_value = values.mean(dim=1, keepdim=True)
                uniform_cross = F.linear(
                    uniform_value, pool.cross_mha.out_proj.weight, pool.cross_mha.out_proj.bias
                )
                actual_pool = pool.norm2(q + actual_cross).squeeze(1)
                uniform_pool = pool.norm2(q + uniform_cross).squeeze(1)
                centroid = c.mean(dim=1, keepdim=True)
                entropy = -(average_weights.clamp_min(1e-12)
                            * average_weights.clamp_min(1e-12).log()).sum()
                entropy /= np.log(average_weights.numel())
                row = collected[variant]
                row["projected_pairwise_cosine"].extend(pairwise_cosine(c))
                row["contextualized_pairwise_cosine"].extend(pairwise_cosine(contextualized))
                row["value_pairwise_cosine"].extend(pairwise_cosine(values))
                row["attention_normalized_entropy"].append(entropy.item())
                row["attention_max"].append(average_weights.max().item())
                row["attended_vs_uniform_pool_relative_l2"].append(
                    relative_delta(actual_pool, uniform_pool)
                )
                row["projected_spread_to_centroid_ratio"].append(
                    ((c - centroid).norm(dim=-1).mean()
                     / centroid.norm(dim=-1).mean().clamp_min(1e-12)).item()
                )

    final_bias = pool.projector[2].bias.float()
    checkpoint_deltas = {}
    if options.trainer_checkpoint.exists():
        with safe_open(options.trainer_checkpoint, framework="pt", device="cpu") as file:
            for name in (
                "embds_pool.projector.0.weight", "embds_pool.projector.0.bias",
                "embds_pool.projector.2.weight", "embds_pool.projector.2.bias",
            ):
                earlier = file.get_tensor(f"hypernet.{name}").float()
                final = state[name].float()
                checkpoint_deltas[name] = relative_delta(final, earlier)

    result = {
        "sample_count": count,
        "parameters": {
            "first_linear_weight": matrix_report(pool.projector[0].weight),
            "final_linear_weight": matrix_report(pool.projector[2].weight),
            "first_bias_l2": pool.projector[0].bias.float().norm().item(),
            "final_bias_l2": final_bias.norm().item(),
            "checkpoint_2400_to_saved_final_relative_delta": checkpoint_deltas,
        },
        "ablations": {
            variant: {name: summary(values) for name, values in metrics.items()}
            for variant, metrics in collected.items()
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
