from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from data_utils import FEATURE_COLUMNS, LABEL_COLUMN, TARGET_COLUMN, labels_from_fos, load_slope_excel


CLASS_LABELS = ["unstable", "critical", "stable"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a random forest FOS model.")
    parser.add_argument("--train", default="outputs/training_data.csv", help="Generated training CSV.")
    parser.add_argument("--test", default="JIAxin WANG_create.xlsx", help="Excel test dataset.")
    parser.add_argument("--outputs", default="outputs", help="Output directory.")
    parser.add_argument("--n-estimators", type=int, default=500, help="Random forest tree count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def train_and_evaluate(args: argparse.Namespace) -> dict:
    output_dir = Path(args.outputs)
    figure_dir = output_dir / "figures"
    model_dir = output_dir / "model"
    figure_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train)
    test_df = load_slope_excel(args.test)

    train_df = train_df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].to_numpy()

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)

    true_labels = labels_from_fos(y_test)
    predicted_labels = labels_from_fos(predicted)

    predictions = test_df.copy()
    predictions["Predicted_FOS"] = predicted
    predictions["True_Label"] = true_labels
    predictions["Predicted_Label"] = predicted_labels
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    rmse = float(np.sqrt(mean_squared_error(y_test, predicted)))
    metrics = {
        "regression": {
            "MAE": float(mean_absolute_error(y_test, predicted)),
            "RMSE": rmse,
            "R2": float(r2_score(y_test, predicted)),
        },
        "classification": {
            "accuracy": float(accuracy_score(true_labels, predicted_labels)),
            "report": classification_report(
                true_labels,
                predicted_labels,
                labels=CLASS_LABELS,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                true_labels,
                predicted_labels,
                labels=CLASS_LABELS,
            ).tolist(),
            "labels": CLASS_LABELS,
        },
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    joblib.dump(model, model_dir / "random_forest_fos.joblib")
    _save_figures(y_test, predicted, true_labels, predicted_labels, model, figure_dir)
    return metrics


def _save_figures(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    model: RandomForestRegressor,
    figure_dir: Path,
) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.75, edgecolor="none")
    lower = float(min(np.min(y_true), np.min(y_pred)))
    upper = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([lower, upper], [lower, upper], color="crimson", linewidth=2)
    plt.xlabel("Actual FOS")
    plt.ylabel("Predicted FOS")
    plt.title("Predicted vs Actual Safety Factor")
    plt.tight_layout()
    plt.savefig(figure_dir / "predicted_vs_actual.png", dpi=180)
    plt.close()

    residuals = y_pred - y_true
    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=28, color="#4C78A8", edgecolor="white")
    plt.axvline(0, color="crimson", linewidth=2)
    plt.xlabel("Residual (Predicted - Actual)")
    plt.ylabel("Count")
    plt.title("Prediction Residuals")
    plt.tight_layout()
    plt.savefig(figure_dir / "residuals.png", dpi=180)
    plt.close()

    matrix = confusion_matrix(true_labels, predicted_labels, labels=CLASS_LABELS)
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks(range(len(CLASS_LABELS)), CLASS_LABELS, rotation=30, ha="right")
    plt.yticks(range(len(CLASS_LABELS)), CLASS_LABELS)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(figure_dir / "confusion_matrix.png", dpi=180)
    plt.close()

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    plt.figure(figsize=(7, 5))
    importances.plot(kind="barh", color="#59A14F")
    plt.xlabel("Importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(figure_dir / "feature_importance.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    metrics = train_and_evaluate(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

