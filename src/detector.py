"""Forecasting-based anomaly detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


@dataclass
class WindowSample:
    x: torch.Tensor
    y: torch.Tensor
    start_idx: int


class ForecastingDataset(Dataset):
    """
    Sliding-window dataset for forecasting anomaly detection.

    Input:
        x: [context_len, num_features]
        y: [prediction_len, num_features]
    """

    def __init__(
        self,
        df: pd.DataFrame,
        context_len: int,
        prediction_len: int,
    ):
        if "is_anomaly" not in df.columns:
            raise ValueError("Dataset must contain 'is_anomaly' column.")

        self.context_len = context_len
        self.prediction_len = prediction_len

        self.feature_columns = [c for c in df.columns if c != "is_anomaly"]
        self.values = df[self.feature_columns].values.astype(np.float32)

        self.length = len(self.values) - context_len - prediction_len + 1
        if self.length <= 0:
            raise ValueError(
                f"Time series is too short: len={len(self.values)}, "
                f"context_len={context_len}, prediction_len={prediction_len}"
            )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        x = self.values[idx : idx + self.context_len]
        y = self.values[
            idx + self.context_len : idx + self.context_len + self.prediction_len
        ]

        return {
            "input": torch.tensor(x, dtype=torch.float32),
            "target": torch.tensor(y, dtype=torch.float32),
            "start_idx": idx,
        }


class AnomalyDetector:
    """Detector for forecasting foundation models."""

    def __init__(
        self,
        model,
        context_len: int,
        prediction_len: int = 1,
        device: str = "cpu",
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        self.model = model
        self.context_len = context_len
        self.prediction_len = prediction_len
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset",
        normalize: bool = True,
    ) -> np.ndarray:
        dataset = ForecastingDataset(
            df=df,
            context_len=self.context_len,
            prediction_len=self.prediction_len,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        anomaly_score = torch.zeros(len(df), dtype=torch.float32, device=self.device)
        counts = torch.zeros(len(df), dtype=torch.float32, device=self.device)

        for batch in tqdm(dataloader, desc=f"Predicting {dataset_name}", leave=False):
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device)
            start_indices = batch["start_idx"]

            with torch.no_grad():
                predictions = self.model.predict(inputs)

            errors = ((targets - predictions) ** 2).mean(dim=2)

            for i, start_idx in enumerate(start_indices):
                pred_start = start_idx + self.context_len
                pred_end = pred_start + self.prediction_len

                anomaly_score[pred_start:pred_end] += errors[i]
                counts[pred_start:pred_end] += 1.0

        valid_mask = counts > 0
        anomaly_score[valid_mask] = anomaly_score[valid_mask] / counts[valid_mask]

        scores = anomaly_score.cpu().numpy()

        if normalize:
            max_val = np.nanmax(scores)
            if max_val > 0:
                scores = scores / max_val

        scores = np.nan_to_num(scores, nan=0.0)
        return scores


def save_predictions(
    anomaly_scores: np.ndarray,
    prediction_path: str,
    index: pd.Index | None = None,
    score_name: str = "anomaly_score",
) -> None:
    series = pd.Series(anomaly_scores, index=index, name=score_name)
    series.to_csv(prediction_path)