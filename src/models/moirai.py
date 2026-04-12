"""Moirai model implementation."""

from __future__ import annotations

import torch
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

from models.base import BaseAnomalyModel, ModelType


class MoiraiModel(BaseAnomalyModel):
    """Moirai forecasting model."""

    def __init__(
        self,
        model_path: str = "Salesforce/moirai-1.0-R-base",
        context_len: int = 1000,
        prediction_len: int = 1,
        patch_size: str = "auto",
        num_samples: int = 100,
        **kwargs,
    ):
        super().__init__(
            model_type=ModelType.FORECASTING,
            context_len=context_len,
            prediction_len=prediction_len,
            **kwargs,
        )
        self.model_path = model_path
        self.patch_size = patch_size
        self.num_samples = num_samples

    def load_model(self, model_path: str | None = None, **kwargs) -> None:
        if model_path is not None:
            self.model_path = model_path

        module = MoiraiModule.from_pretrained(self.model_path)
        self.model = MoiraiForecast(
            module=module,
            prediction_length=self.prediction_len,
            context_length=self.context_len,
            patch_size=self.patch_size,
            num_samples=self.num_samples,
            target_dim=kwargs.get("target_dim", 1),
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        self.model = self.model.to(self.device).eval()

    def predict(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Model is not loaded. Call load_model() first.")

        batch_size, seq_len, num_features = inputs.shape

        # Moirai expects additional dummy first step
        past_target = torch.cat(
            [
                torch.zeros(
                    batch_size, 1, num_features,
                    device=self.device, dtype=inputs.dtype
                ),
                inputs,
            ],
            dim=1,
        )

        past_observed_target = torch.ones(
            batch_size, seq_len + 1, num_features,
            device=self.device, dtype=torch.bool
        )
        past_observed_target[:, 0] = False

        past_is_pad = torch.zeros(batch_size, seq_len + 1, device=self.device)
        past_is_pad[:, 0] = 1

        with torch.no_grad():
            model_output = self.model.forward(
                past_target=past_target,
                past_observed_target=past_observed_target,
                past_is_pad=past_is_pad,
            )

        # (B, num_samples, prediction_len, F) -> (B, prediction_len, F)
        predictions = model_output.mean(dim=1)[:, : self.prediction_len]

        return predictions