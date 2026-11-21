"""Reproduce and extend the 2024 NPHA healthcare-utilization classification study.

Usage:
    python train.py --data data/NPHA-doctor-visits.csv

The original paper is docs/original_report.pdf. This script keeps its two model
families, features, and 70/15/15 split but adds a baseline and simple graphics.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

TARGET = "Number_of_Doctors_Visited"
TARGET_LABELS = [1, 2, 3]


def normalize_column_names(columns):
    """Accept both whitespace and underscore variants found in NPHA copies."""
    normalized = []
    for col in columns:
        value = re.sub(r"[^0-9A-Za-z]+", "_", str(col)).strip("_")
        value = re.sub(r"_+", "_", value)
        value = value.replace("Phyiscal_Health", "Physical_Health")
        value = value.replace("Uknown_", "Unknown_")
        normalized.append(value)
    return normalized


def prepare_data(dataframe: pd.DataFrame):
    """Apply the report's feature selection and refused-response filtering."""
    df = dataframe.copy()
    df.columns = normalize_column_names(df.columns)
    required = {
        TARGET,
        "Prescription_Sleep_Medication",
        "Employment",
        "Dental_Health",
        "Mental_Health",
        "Physical_Health",
        "Race",
        "Trouble_Sleeping",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {', '.join(missing)}")

    # The original notebook removed refused values in these survey fields.
    for col in [
        "Prescription_Sleep_Medication",
        "Employment",
        "Dental_Health",
        "Mental_Health",
        "Physical_Health",
    ]:
        df = df[df[col] != -1]
    df = df[df["Race"] != -2]
    df = df.dropna()

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET, "Trouble_Sleeping", "Race"])
    X = X.apply(pd.to_numeric, errors="raise")
    return X, y


def perform_analysis(dataframe: pd.DataFrame, output_dir: Path, *, plot: bool = True):
    X, y = prepare_data(dataframe)
    X_train, X_remainder, y_train, y_remainder = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_remainder, y_remainder, test_size=0.5, random_state=42
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
    }
    metrics = {
        "samples_before_cleaning": int(len(dataframe)),
        "samples_after_cleaning": int(len(X)),
        "features": list(X.columns),
        "split_sizes": {
            "train": len(X_train), "validation": len(X_val), "test": len(X_test)
        },
        "models": {},
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        scores = {
            "training_accuracy": float(accuracy_score(y_train, model.predict(X_train))),
            "validation_accuracy": float(accuracy_score(y_val, model.predict(X_val))),
            "test_accuracy": float(accuracy_score(y_test, model.predict(X_test))),
        }
        metrics["models"][name] = scores
        print(f"\n{name.replace('_', ' ').title()}")
        for key, value in scores.items():
            print(f"  {key.replace('_', ' ')}: {value:.2%}")

    # The report chose logistic regression based on validation performance.
    chosen_model = models["logistic_regression"]
    test_pred = chosen_model.predict(X_test)
    print("\nLogistic regression: test-set classification report")
    print(classification_report(y_test, test_pred, labels=TARGET_LABELS, zero_division=0))
    metrics["logistic_test_classification_report"] = classification_report(
        y_test,
        test_pred,
        labels=TARGET_LABELS,
        zero_division=0,
        output_dict=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    if plot:
        cm = confusion_matrix(y_test, test_pred, labels=TARGET_LABELS)
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["0–1", "2–3", "4+"],
        )
        display.plot(cmap="Blues", colorbar=False)
        display.ax_.set_title("Logistic regression — held-out test set")
        display.ax_.set_xlabel("Predicted doctors visited")
        display.ax_.set_ylabel("Actual doctors visited")
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png", dpi=180)
        plt.close()

        y.value_counts().sort_index().plot(kind="bar")
        plt.xlabel("Number of doctors visited (class)")
        plt.ylabel("Respondents")
        plt.title("NPHA target distribution after filtering")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / "class_distribution.png", dpi=180)
        plt.close()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/NPHA-doctor-visits.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(
            f"Cannot find dataset at {args.data}. Download NPHA-doctor-visits.csv "
            "from https://archive.ics.uci.edu/dataset/936/national+poll+on+healthy+aging+npha"
        )
    df = pd.read_csv(args.data)
    perform_analysis(df, args.output_dir)
    print(f"\nSaved metrics and plots to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
