"""Base classes and interfaces for anomaly detection models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional

import torch


class ModelType(Enum):
    """Supported anomaly detection model types."""

    FORECASTING = "forecasting"


class BaseAnomalyModel(ABC):
    """
    Base class for forecasting-based anomaly detection models.

    All models must implement:
    - load_model()
    - predict()
    """

    def __init__(
        self,
        model_type: ModelType = ModelType.FORECASTING,
        device: str = "cpu",
        context_len: int = 512,
        prediction_len: int = 1,
        batch_size: int = 32,
        **kwargs,
    ):
        self.model_type = model_type
        self.device = device
        self.context_len = context_len
        self.prediction_len = prediction_len
        self.batch_size = batch_size
        self.model = None
        self.config = kwargs

    @abstractmethod
    def load_model(self, model_path: str | None = None, **kwargs) -> None:
        """Load pretrained model."""
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        inputs: torch.Tensor,
        batch: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Predict future values.

        Args:
            inputs: tensor of shape (batch_size, context_len, num_features)

        Returns:
            tensor of shape (batch_size, prediction_len, num_features)
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_type={self.model_type.value}, "
            f"device={self.device}, "
            f"context_len={self.context_len}, "
            f"prediction_len={self.prediction_len}, "
            f"batch_size={self.batch_size})"
        )