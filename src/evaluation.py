from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from datasets import DatasetManager


def f1_score_custom(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    anomaly_rate: float = 0.05,
    adjust: bool = True,
) -> tuple[float, float, float]:
    y_true = y_true.astype(np.int32)

    gt_aug = np.concatenate([np.zeros(1), y_true, np.zeros(1)]).astype(np.int32)
    gt_diff = gt_aug[1:] - gt_aug[:-1]
    starts = np.where(gt_diff == 1)[0]
    ends = np.where(gt_diff == -1)[0]
    intervals = np.stack([starts, ends], axis=1)

    threshold = np.quantile(y_scores, 1 - anomaly_rate)
    y_pred = (y_scores > threshold).astype(np.int32)

    if adjust:
        for start, end in intervals:
            if y_pred[start:end].sum() > 0:
                y_pred[start:end] = 1

    tp = int((y_true * y_pred).sum())
    fp = int(((1 - y_true) * y_pred).sum())
    fn = int((y_true * (1 - y_pred)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


def evaluate_predictions(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    verbose: bool = True,
) -> dict:
    y_scores = np.nan_to_num(y_scores, nan=0.0)

    roc_auc = roc_auc_score(y_true, y_scores)
    results = {}

    for adjust in [False, True]:
        best_f1 = -1.0
        best_rate = None
        best_metrics = None

        for rate in np.arange(0.001, 0.301, 0.001):
            precision, recall, f1 = f1_score_custom(
                y_true=y_true,
                y_scores=y_scores,
                anomaly_rate=rate,
                adjust=adjust,
            )
            if f1 > best_f1:
                best_f1 = f1
                best_rate = rate
                best_metrics = (precision, recall, f1)

        key = "point-adjustment" if adjust else "no-point-adjustment"
        results[key] = {
            "anomaly_rate": float(best_rate),
            "precision": float(best_metrics[0]),
            "recall": float(best_metrics[1]),
            "F1-score": float(best_metrics[2]),
            "ROC_AUC": float(roc_auc),
        }

        if verbose:
            title = "with point adjustment" if adjust else "without point adjustment"
            print(f"Best F1-score {title}")
            print(
                f"anomaly rate: {best_rate:.3f} | "
                f"precision: {best_metrics[0]:.5f} | "
                f"recall: {best_metrics[1]:.5f} | "
                f"F1-score: {best_metrics[2]:.5f} | "
                f"ROC AUC: {roc_auc:.5f}\n"
            )

    return results


def evaluate_collection(
    prediction_dir: str | Path,
    collection: str,
    data_path: str | Path,
    verbose: bool = True,
) -> dict:
    prediction_dir = Path(prediction_dir)
    dm = DatasetManager(data_path)

    y_true_all = []
    y_scores_all = []

    for _, dataset_name in dm.list_all(collection=collection):
        pred_file = prediction_dir / f"{collection}-{dataset_name}.csv"

        if not pred_file.exists():
            if verbose:
                print(f"Warning: missing prediction for {collection}/{dataset_name}")
            continue

        df_test = dm.get_test_dataset(collection, dataset_name)
        y_true = df_test["is_anomaly"].values.astype(np.int32)

        y_scores = pd.read_csv(pred_file, index_col=0)["anomaly_score"].values
        y_scores = np.nan_to_num(y_scores, nan=0.0)

        if len(y_true) != len(y_scores):
            raise ValueError(
                f"Length mismatch for {collection}/{dataset_name}: "
                f"y_true={len(y_true)}, y_scores={len(y_scores)}"
            )

        y_true_all.append(y_true)
        y_scores_all.append(y_scores)

    if not y_true_all:
        raise ValueError(f"No prediction files found for collection {collection}")

    y_true_concat = np.concatenate(y_true_all)
    y_scores_concat = np.concatenate(y_scores_all)

    return evaluate_predictions(y_true_concat, y_scores_concat, verbose=verbose)


def save_evaluation_results(results: dict, output_path: str | Path) -> None:
    rows = []
    for mode, metrics in results.items():
        row = {"mode": mode}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)