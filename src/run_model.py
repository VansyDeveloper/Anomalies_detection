from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from datasets import DatasetManager
from detector import AnomalyDetector, save_predictions
from evaluation import evaluate_collection, save_evaluation_results
from model_factory import create_model


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "is_anomaly" not in df.columns:
        raise ValueError("Input DataFrame must contain 'is_anomaly' column.")

    df = df.copy()
    feature_columns = [c for c in df.columns if c != "is_anomaly"]

    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns].astype(float))

    return df


def run_collection(
    config: dict,
    collection: str,
    data_path: str | Path,
    prediction_root: str | Path,
    results_root: str | Path,
) -> None:
    model_name = config["model_name"]
    device = config.get("device", "cpu")

    print(f"Initializing {model_name} on {device}...")

    model_kwargs = dict(config)
    model_kwargs.pop("model_name", None)

    model = create_model(model_name, **model_kwargs)
    model.load_model()

    dm = DatasetManager(data_path)
    datasets = dm.list_datasets(collection)

    print(f"Processing {len(datasets)} dataset(s)...")

    prediction_dir = Path(prediction_root) / model_name / collection
    prediction_dir.mkdir(parents=True, exist_ok=True)

    detector = AnomalyDetector(
        model=model,
        context_len=model.context_len,
        prediction_len=model.prediction_len,
        device=device,
        batch_size=getattr(model, "batch_size", config.get("batch_size", 32)),
    )

    for dataset_name in tqdm(datasets, desc=collection):
        df_test = dm.get_test_dataset(collection, dataset_name)
        df_test = preprocess_dataframe(df_test)

        anomaly_scores = detector.detect_anomalies(
            df=df_test,
            dataset_name=f"{collection}/{dataset_name}",
            normalize=True,
        )

        pred_file = prediction_dir / f"{collection}-{dataset_name}.csv"
        save_predictions(
            anomaly_scores=anomaly_scores,
            prediction_path=pred_file,
            index=df_test.index,
        )

    print(f"\nResults saved to: {prediction_dir}")

    print(f"\nEvaluating {collection}...\n")
    results = evaluate_collection(
        prediction_dir=prediction_dir,
        collection=collection,
        data_path=data_path,
        verbose=True,
    )

    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    output_file = results_root / f"{model_name}_{collection}_eval.csv"
    save_evaluation_results(results, output_file)

    print(f"Evaluation saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run anomaly detection with foundation models"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--collection", type=str, required=True, help="Collection name")
    parser.add_argument(
        "--data-path",
        type=str,
        default="./timeeval_data",
        help="Path to TimeEval datasets",
    )
    parser.add_argument(
        "--prediction-root",
        type=str,
        default="./predictions",
        help="Directory for prediction CSVs",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="./results",
        help="Directory for evaluation CSVs",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    run_collection(
        config=config,
        collection=args.collection,
        data_path=args.data_path,
        prediction_root=args.prediction_root,
        results_root=args.results_root,
    )


if __name__ == "__main__":
    main()