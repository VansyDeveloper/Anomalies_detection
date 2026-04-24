from __future__ import annotations

import torch

if not hasattr(torch.mps, "is_available"):
    torch.mps.is_available = lambda: False

import timesfm

from models.base import BaseAnomalyModel, ModelType


class TimesFMModel(BaseAnomalyModel):

    def __init__(
        self,
        model_path: str = "google/timesfm-2.5-200m-pytorch",
        context_len: int = 512,
        prediction_len: int = 1,
        **kwargs,
    ):
        super().__init__(
            model_type=ModelType.FORECASTING,
            context_len=context_len,
            prediction_len=prediction_len,
            **kwargs,
        )
        self.model_path = model_path

    def load_model(self, model_path: str | None = None, **kwargs) -> None:
        if model_path is not None:
            self.model_path = model_path

        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            self.model_path,
            device=self.device,
        )

        self.model.compile(
            timesfm.ForecastConfig(
                max_context=self.context_len,
                max_horizon=self.prediction_len,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=False,
                per_core_batch_size=self.batch_size,
            )
        )

    def predict(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Model is not loaded. Call load_model() first.")

        batch_size, context_len, num_features = inputs.shape

        max_context = self.model.forecast_config.max_context

        padded_inputs = torch.zeros(
            batch_size, max_context, num_features,
            device=self.device, dtype=inputs.dtype
        )
        padded_inputs[:, -context_len:, :] = inputs

        masks = torch.ones(
            batch_size, max_context, num_features,
            device=self.device, dtype=torch.bool
        )
        masks[:, -context_len:, :] = False

        inputs_reshaped = padded_inputs.transpose(1, 2).reshape(
            batch_size * num_features, max_context
        )
        masks_reshaped = masks.transpose(1, 2).reshape(
            batch_size * num_features, max_context
        )

        with torch.no_grad():
            point_forecast, _ = self.model.compiled_decode(
                horizon=self.prediction_len,
                inputs=inputs_reshaped,
                masks=masks_reshaped,
            )

        predictions = point_forecast.reshape(
            batch_size, num_features, self.prediction_len
        ).transpose(1, 2)

        return predictions