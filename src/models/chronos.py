from __future__ import annotations

import torch
from chronos import BaseChronosPipeline, ChronosBoltPipeline

from models.base import BaseAnomalyModel, ModelType


class ChronosModel(BaseAnomalyModel):

    def __init__(
        self,
        model_path: str = "amazon/chronos-t5-small",
        context_len: int = 1000,
        prediction_len: int = 1,
        num_samples: int = 1,
        **kwargs,
    ):
        super().__init__(
            model_type=ModelType.FORECASTING,
            context_len=context_len,
            prediction_len=prediction_len,
            **kwargs,
        )
        self.model_path = model_path
        self.num_samples = num_samples

    def load_model(self, model_path: str = None, **kwargs):
        if model_path:
            self.model_path = model_path

        device_map = "cpu" if self.device == "mps" else self.device
        dtype = torch.float32

        self.model = BaseChronosPipeline.from_pretrained(
            self.model_path,
            device_map=device_map,
            torch_dtype=dtype,
        )

    def predict(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        batch_size, context_len, num_features = inputs.shape

        inputs_reshaped = (
            inputs.permute(0, 2, 1)
            .reshape(batch_size * num_features, context_len)
            .detach()
            .to("cpu")
        )

        with torch.no_grad():
            predictions_bf_ns_lp = self.model.predict(
                inputs_reshaped,
                prediction_length=self.prediction_len,
            )

        if isinstance(self.model, ChronosBoltPipeline):
            predictions_bf_lp = predictions_bf_ns_lp[:, 4]
        else:
            predictions_bf_lp = predictions_bf_ns_lp.mean(dim=1)

        batch_predictions = predictions_bf_lp.reshape(
            batch_size, num_features, self.prediction_len
        )

        predictions = batch_predictions.permute(0, 2, 1)

        return predictions.to(inputs.device)