"""Plot training curves recorded by the Embd2Adapter experiments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


MEAN_EXPERIMENT = "HyperNetTrainer-TypeMeanEmbedding"
QDIFF_EXPERIMENT = "HyperNetTrainer-TypeCrossAttention"


def load_training_history(
    result_path: str | Path,
    experiment: str,
) -> list[dict]:
    """Load the unique training-history record for an experiment."""
    result_path = Path(result_path)
    with result_path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    matches = [
        record
        for record in records
        if record.get("record_type") == "training_history"
        and record.get("experiment") == experiment
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {experiment!r} training record in {result_path}, "
            f"but found {len(matches)}."
        )
    return matches[0]["log_history"]


def extract_step_metric(
    log_history: list[dict],
    metric: str,
) -> tuple[list[int], list[float]]:
    """Extract steps and values from log entries containing a metric."""
    points = [
        (int(entry["step"]), float(entry[metric]))
        for entry in log_history
        if metric in entry and "step" in entry
    ]
    if not points:
        raise ValueError(f"Metric {metric!r} was not found in the log history.")
    steps, values = zip(*points)
    return list(steps), list(values)


def _plot_comparison(
    mean_history: list[dict],
    qdiff_history: list[dict],
    metric: str,
    title: str,
    ylabel: str,
):
    figure, axis = plt.subplots(figsize=(9, 5))
    for label, history, marker in (
        ("MeanEmbds", mean_history, "o"),
        ("QueryDiffPooling", qdiff_history, "s"),
    ):
        steps, values = extract_step_metric(history, metric)
        axis.plot(
            steps,
            values,
            marker=marker,
            markersize=4,
            linewidth=1.8,
            label=label,
        )

    axis.set_title(title)
    axis.set_xlabel("Training step")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure, axis


def plot_training_comparison(
    mean_result_path: str | Path,
    qdiff_result_path: str | Path,
    save_dir: str | Path | None = None,
    show: bool = True,
):
    """Draw separate loss and validation-token-F1 comparison figures."""
    mean_history = load_training_history(mean_result_path, MEAN_EXPERIMENT)
    qdiff_history = load_training_history(qdiff_result_path, QDIFF_EXPERIMENT)

    loss_figure, loss_axis = _plot_comparison(
        mean_history,
        qdiff_history,
        metric="loss",
        title="Training Loss: MeanEmbds vs QueryDiffPooling",
        ylabel="Training loss",
    )
    accuracy_figure, accuracy_axis = _plot_comparison(
        mean_history,
        qdiff_history,
        metric="eval_token_f1",
        title="Validation Accuracy: MeanEmbds vs QueryDiffPooling",
        ylabel="Validation token F1",
    )

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        loss_figure.savefig(
            save_dir / "training_loss_comparison.png",
            dpi=200,
            bbox_inches="tight",
        )
        accuracy_figure.savefig(
            save_dir / "validation_accuracy_comparison.png",
            dpi=200,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    return (loss_figure, loss_axis), (accuracy_figure, accuracy_axis)
