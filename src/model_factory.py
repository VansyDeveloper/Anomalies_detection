from __future__ import annotations

from typing import Any


def create_model(model_name: str, **kwargs: Any):
    model_name = model_name.lower()

    if model_name == "chronos":
        from models.chronos import ChronosModel
        return ChronosModel(**kwargs)

    if model_name == "moirai":
        from models.moirai import MoiraiModel
        return MoiraiModel(**kwargs)

    if model_name == "timesfm":
        from models.timesfm import TimesFMModel
        return TimesFMModel(**kwargs)

    if model_name == "ttm":
        from models.ttm import TTMModel
        return TTMModel(**kwargs)

    raise ValueError(f"Unknown model: {model_name}")