from __future__ import annotations

import torch

from models.base import BaseAnomalyModel, ModelType


class TTMModel(BaseAnomalyModel):
    def __init__(
        self,
        model_path: str = "ibm-granite/granite-timeseries-ttm-r2",
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

        from tsfm_public import TinyTimeMixerForPrediction

        self.model = TinyTimeMixerForPrediction.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
        )
        self.model = self.model.to(self.device).eval()

    def predict(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Model is not loaded. Call load_model() first.")

        batch_size, context_len, num_features = inputs.shape

        model_context_len = 512
        padded_inputs = torch.zeros(
            batch_size, model_context_len, num_features,
            device=self.device, dtype=inputs.dtype
        )
        padded_inputs[:, -context_len:, :] = inputs

        with torch.no_grad():
            model_output = self.model.forward(padded_inputs)

        predictions = model_output.prediction_outputs[:, : self.prediction_len, :]
        return predictions