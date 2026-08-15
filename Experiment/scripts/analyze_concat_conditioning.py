"""Measure contribution of pooled context, layer ID, and position ID conditioning."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Experiment"))

from src.Embd2Adapter.Type_CrossAttention.hypernet_crossattn_models import (  # noqa: E402
    HyperNetController_CrossAttention,
)


def relative_delta(left, right):
    return ((left.float() - right.float()).norm()
            / right.float().norm().clamp_min(1e-12)).item()


def summarize(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(mean=float(a.mean()), std=float(a.std()), min=float(a.min()),
                median=float(np.median(a)), max=float(a.max()))


def flatten_adapter(outputs):
    tensors = []
    for adapter in outputs:
        for sampler in adapter:
            tensors.extend((sampler.weight.reshape(-1), sampler.bias.reshape(-1)))
    return torch.cat(tensors).float()


def main():
    cache_path = ROOT / "Notebooks/outputs/cache/embeddings_cache_validation.npy"
    checkpoint = ROOT / "Notebooks/outputs/cross_attn/hypernet_state_dict.pt"
    output = ROOT / "Experiment/Result/concat_conditioning_diagnostics.json"
    cache = np.load(cache_path, allow_pickle=True).item()
    query_cache = np.load(
        cache_path.with_name(f"{cache_path.stem}_queries{cache_path.suffix}"),
        allow_pickle=True,
    ).item()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    embd_dim = int(cache["embedding_dim"])
    projected = state["up_sampler_hyper_net.weight_generator.weight"].shape[1]
    bottleneck = state["up_sampler_hyper_net.bias_generator.weight"].shape[0]
    input_dim = state["up_sampler_hyper_net.weight_generator.weight"].shape[0] // bottleneck
    model = HyperNetController_CrossAttention(
        device=torch.device("cpu"), embd_dim=embd_dim,
        projected_embd_dim=projected, input_dim=input_dim,
        reduction_factor=input_dim // bottleneck,
        task_hidden_dim=state["vec_hypernet.task_embedding_generator.0.weight"].shape[0],
        num_layers=state["layer_id_embeddings.weight"].shape[0],
    )
    model.load_state_dict(state)
    model.eval()
    rng = np.random.default_rng(33)
    indices = rng.choice(len(cache["embeddings"]), size=100, replace=False)
    layers = sorted({0, model.num_layers // 2, model.num_layers - 1})
    metrics = {key: [] for key in (
        "pooled_context_norm", "layer_embedding_norm", "position_embedding_norm",
        "task_delta_without_context", "task_delta_without_layer",
        "task_delta_without_position", "adapter_delta_without_context",
        "adapter_delta_without_layer", "adapter_delta_without_position",
    )}

    def task_vector(context, layer_vector, position_vector):
        joined = torch.cat((context.reshape(-1), layer_vector, position_vector))
        return model.vec_hypernet(joined)

    with torch.inference_mode():
        for index in indices:
            contexts = torch.as_tensor(
                np.asarray(cache["embeddings"][index]), dtype=torch.float32
            ).unsqueeze(0)
            query = torch.as_tensor(
                np.asarray(query_cache["embeddings"][index]), dtype=torch.float32
            ).unsqueeze(0)
            pooled = model.pool_embeddings(contexts, query)
            metrics["pooled_context_norm"].append(pooled.norm().item())
            for layer in layers:
                layer_vector = model.layer_id_embeddings.weight[layer]
                metrics["layer_embedding_norm"].append(layer_vector.norm().item())
                for position in (0, 1):
                    position_vector = model.position_id_embeddings.weight[position]
                    metrics["position_embedding_norm"].append(position_vector.norm().item())
                    baseline_task = task_vector(pooled, layer_vector, position_vector)
                    variants = {
                        "context": task_vector(torch.zeros_like(pooled), layer_vector, position_vector),
                        "layer": task_vector(pooled, torch.zeros_like(layer_vector), position_vector),
                        "position": task_vector(pooled, layer_vector, torch.zeros_like(position_vector)),
                    }
                    for name, variant in variants.items():
                        metrics[f"task_delta_without_{name}"].append(
                            relative_delta(variant, baseline_task)
                        )
                baseline_adapter = flatten_adapter(
                    model(pooled_embedding=pooled, layer_id=layer)
                )
                # Adapter-level ablations require temporarily replacing ID rows.
                saved_layer = model.layer_id_embeddings.weight[layer].clone()
                model.layer_id_embeddings.weight[layer].zero_()
                without_layer = flatten_adapter(
                    model(pooled_embedding=pooled, layer_id=layer)
                )
                model.layer_id_embeddings.weight[layer].copy_(saved_layer)
                saved_positions = model.position_id_embeddings.weight.clone()
                model.position_id_embeddings.weight.zero_()
                without_position = flatten_adapter(
                    model(pooled_embedding=pooled, layer_id=layer)
                )
                model.position_id_embeddings.weight.copy_(saved_positions)
                without_context = flatten_adapter(
                    model(pooled_embedding=torch.zeros_like(pooled), layer_id=layer)
                )
                metrics["adapter_delta_without_context"].append(
                    relative_delta(without_context, baseline_adapter)
                )
                metrics["adapter_delta_without_layer"].append(
                    relative_delta(without_layer, baseline_adapter)
                )
                metrics["adapter_delta_without_position"].append(
                    relative_delta(without_position, baseline_adapter)
                )

    result = {
        "sample_count": 100,
        "layer_ids": layers,
        "metrics": {name: summarize(values) for name, values in metrics.items()},
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
