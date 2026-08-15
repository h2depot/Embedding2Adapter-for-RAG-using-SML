"""Inspect a pretrained GatedCrossAttentionPooling state dict."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Experiment.src.Embd2Adapter.Type_CrossAttention.hypernet_crossattn_models import (  # noqa: E402
    GatedCrossAttentionPooling,
)


def summary(values):
    array = np.asarray(values, dtype=np.float64)
    return dict(mean=float(array.mean()), std=float(array.std()),
                min=float(array.min()), median=float(np.median(array)),
                max=float(array.max()))


def pairwise_cosine(vectors):
    vectors = F.normalize(vectors.float(), dim=-1)
    matrix = vectors @ vectors.transpose(-1, -2)
    mask = ~torch.eye(matrix.shape[-1], dtype=torch.bool)
    return matrix[0][mask].tolist()


def js(left, right):
    mixture = 0.5 * (left + right)
    return 0.5 * (
        (left * (left.clamp_min(1e-12).log() - mixture.clamp_min(1e-12).log())).sum()
        + (right * (right.clamp_min(1e-12).log() - mixture.clamp_min(1e-12).log())).sum()
    )


def main():
    cache_path = ROOT / "Notebooks/outputs/cache/embeddings_cache_validation.npy"
    state_path = ROOT / "Notebooks/outputs/cross_attn/attention_pretrain_state_dict.pt"
    output_path = ROOT / "Experiment/Result/gated_pooling_diagnostics.json"
    contexts_cache = np.load(cache_path, allow_pickle=True).item()["embeddings"]
    query_cache = np.load(
        cache_path.with_name(f"{cache_path.stem}_queries{cache_path.suffix}"),
        allow_pickle=True,
    ).item()["embeddings"]
    model = GatedCrossAttentionPooling(embd_dim=1024)
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    model.eval()
    rng = np.random.default_rng(33)
    indices = rng.choice(len(contexts_cache), size=100, replace=False)
    raw_queries, projected_queries = [], []
    metrics = {key: [] for key in (
        "raw_context_pairwise_cosine", "projected_context_pairwise_cosine",
        "refined_context_pairwise_cosine", "self_attention_update_to_context_ratio",
        "gated_residual_to_context_ratio", "self_gate_mean_absolute",
        "self_gate_saturation_fraction", "post_ffn_pairwise_cosine",
        "ffn_update_to_input_ratio", "attention_entropy", "attention_max",
        "shuffled_query_attention_l1", "shuffled_query_attention_js",
        "shuffled_query_top1_changed",
    )}
    with torch.inference_mode():
        for position, index in enumerate(indices):
            raw_contexts = torch.as_tensor(
                np.asarray(contexts_cache[index]), dtype=torch.float32
            ).unsqueeze(0)
            raw_query = torch.as_tensor(
                np.asarray(query_cache[index]), dtype=torch.float32
            ).unsqueeze(0)
            projected_contexts = model.context_projector(raw_contexts)
            projected_query = model.query_projector(raw_query)
            update, _ = model.self_attention.mha(
                projected_contexts, projected_contexts, projected_contexts,
                need_weights=False,
            )
            gate_features = torch.cat((
                projected_contexts, update, projected_contexts * update,
                torch.abs(projected_contexts - update),
            ), dim=-1)
            gate = torch.tanh(model.self_attention.gate(gate_features))
            refined = projected_contexts + gate * update
            ffn_update = model.ffn(model.ffn_norm(refined))
            post_ffn = refined + ffn_update
            _, weights = model(raw_contexts, raw_query, True)
            weights = weights.squeeze().float()
            shuffled_index = indices[(position + 1) % len(indices)]
            shuffled_query = torch.as_tensor(
                np.asarray(query_cache[shuffled_index]), dtype=torch.float32
            ).unsqueeze(0)
            _, shuffled_weights = model(raw_contexts, shuffled_query, True)
            shuffled_weights = shuffled_weights.squeeze().float()
            raw_queries.append(raw_query)
            projected_queries.append(projected_query)
            metrics["raw_context_pairwise_cosine"].extend(pairwise_cosine(raw_contexts))
            metrics["projected_context_pairwise_cosine"].extend(pairwise_cosine(projected_contexts))
            metrics["refined_context_pairwise_cosine"].extend(pairwise_cosine(refined))
            metrics["post_ffn_pairwise_cosine"].extend(pairwise_cosine(post_ffn))
            metrics["ffn_update_to_input_ratio"].append(
                (ffn_update.norm() / refined.norm().clamp_min(1e-12)).item()
            )
            metrics["self_attention_update_to_context_ratio"].append(
                (update.norm() / projected_contexts.norm().clamp_min(1e-12)).item()
            )
            metrics["gated_residual_to_context_ratio"].append(
                ((gate * update).norm() / projected_contexts.norm().clamp_min(1e-12)).item()
            )
            metrics["self_gate_mean_absolute"].append(gate.abs().mean().item())
            metrics["self_gate_saturation_fraction"].append(
                (gate.abs() > 0.9).float().mean().item()
            )
            metrics["attention_entropy"].append(
                (-(weights * weights.clamp_min(1e-12).log()).sum()
                 / np.log(weights.numel())).item()
            )
            metrics["attention_max"].append(weights.max().item())
            metrics["shuffled_query_attention_l1"].append(
                (weights - shuffled_weights).abs().sum().item()
            )
            metrics["shuffled_query_attention_js"].append(
                js(weights, shuffled_weights).item()
            )
            metrics["shuffled_query_top1_changed"].append(
                float(weights.argmax() != shuffled_weights.argmax())
            )
    result = {
        "sample_count": len(indices),
        "metrics": {key: summary(values) for key, values in metrics.items()},
        "query_cross_sample_cosine": {
            "raw": summary(pairwise_cosine(torch.cat(raw_queries).unsqueeze(0))),
            "projected": summary(pairwise_cosine(torch.cat(projected_queries).unsqueeze(0))),
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
