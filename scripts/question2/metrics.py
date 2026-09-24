from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def compute_metrics(
    class_labels: Sequence[int],
    class_predictions: Sequence[int],
    regression_labels: Sequence[float],
    regression_predictions: Sequence[float],
) -> dict[str, float | None]:
    """Compute task metrics, representing undefined Pearson correlation as None."""
    labels = np.asarray(class_labels)
    predictions = np.asarray(class_predictions)
    regression_targets = np.asarray(regression_labels, dtype=float)
    regression_outputs = np.asarray(regression_predictions, dtype=float)
    if not (len(labels) == len(predictions) == len(regression_targets) == len(regression_outputs)):
        raise ValueError("Metric inputs must have equal lengths")
    if not len(labels):
        raise ValueError("Metric inputs must not be empty")

    classes = np.union1d(labels, predictions)
    f1_scores = []
    supports = []
    for label in classes:
        true_positive = np.count_nonzero((labels == label) & (predictions == label))
        false_positive = np.count_nonzero((labels != label) & (predictions == label))
        false_negative = np.count_nonzero((labels == label) & (predictions != label))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(2 * true_positive / denominator if denominator else 0.0)
        supports.append(np.count_nonzero(labels == label))

    pearson = None
    if len(regression_targets) > 1 and np.std(regression_targets) and np.std(regression_outputs):
        pearson = float(np.corrcoef(regression_targets, regression_outputs)[0, 1])
    return {
        "accuracy": float(np.mean(labels == predictions)),
        "macro_f1": float(np.mean(f1_scores)),
        "weighted_f1": float(np.average(f1_scores, weights=supports)),
        "mae": float(np.mean(np.abs(regression_targets - regression_outputs))),
        "pearson": pearson,
    }
