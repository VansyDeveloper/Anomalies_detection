"""Utilities for loading TimeEval datasets."""

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from timeeval import DatasetManager as TimeEvalDatasetManager


class DatasetManager:
    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        self.dm = TimeEvalDatasetManager(str(self.data_path))

    def list_collections(self) -> List[str]:
        datasets = self.dm.select()
        return sorted({collection for collection, _ in datasets})

    def list_datasets(self, collection: str) -> List[str]:
        datasets = self.dm.select(collection=collection)
        return sorted([dataset_name for _, dataset_name in datasets])

    def list_all(self, collection: str | None = None) -> List[Tuple[str, str]]:
        if collection is None:
            return list(self.dm.select())
        return list(self.dm.select(collection=collection))

    def get_dataset(
        self,
        collection: str,
        dataset_name: str,
        train: bool = False,
    ) -> pd.DataFrame:
        df = self.dm.get_dataset_df((collection, dataset_name), train=train)

        if "timestamp" in df.columns:
            df = df.set_index("timestamp")

        return df

    def get_test_dataset(self, collection: str, dataset_name: str) -> pd.DataFrame:
        return self.get_dataset(collection, dataset_name, train=False)

    def get_train_dataset(self, collection: str, dataset_name: str) -> pd.DataFrame:
        return self.get_dataset(collection, dataset_name, train=True)