"""Train a confidence-aware gesture classifier from collected landmarks."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from landmark_features import HAND_FEATURE_COUNT, POSE_FEATURE_COUNT


def parse_arguments() -> argparse.Namespace:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train a MediaPipe landmark gesture model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=script_directory / "landmark_data_hand.csv",
        help="Dataset created by collect_landmarks.py",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=script_directory / "landmark_gesture_model.joblib",
        help="Output model bundle",
    )
    parser.add_argument("--confidence", type=float, default=0.80)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--window", type=int, default=12, help="Prediction smoothing frames")
    parser.add_argument("--votes", type=int, default=8, help="Matching frames required")
    arguments = parser.parse_args()
    if not 0.0 < arguments.confidence < 1.0:
        parser.error("--confidence must be between 0 and 1")
    if not 0.0 <= arguments.margin < 1.0:
        parser.error("--margin must be between 0 and 1")
    if arguments.votes < 1 or arguments.votes > arguments.window:
        parser.error("--votes must be between 1 and --window")
    return arguments


def load_dataset(path: Path):
    if not path.exists():
        raise SystemExit(f"Dataset not found: {path}")

    features = []
    labels = []
    sessions = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit("The dataset is empty.")
        feature_names = [name for name in reader.fieldnames if name.startswith("feature_")]
        if not feature_names:
            raise SystemExit("The dataset does not contain landmark feature columns.")
        for row_number, row in enumerate(reader, start=2):
            try:
                vector = [float(row[name]) for name in feature_names]
            except (TypeError, ValueError) as error:
                raise SystemExit(f"Invalid number on row {row_number}: {error}") from error
            features.append(vector)
            labels.append(row["label"].strip().upper())
            sessions.append(row["session"])

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels),
        np.asarray(sessions),
        feature_names,
    )


def make_split(features, labels, sessions):
    sessions_per_label = defaultdict(set)
    for label, session in zip(labels, sessions):
        sessions_per_label[label].add(session)
    minimum_sessions = min(len(values) for values in sessions_per_label.values())

    if minimum_sessions >= 2:
        folds = min(5, minimum_sessions)
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
        train_indices, test_indices = next(splitter.split(features, labels, groups=sessions))
        split_description = f"session-separated validation ({folds} folds)"
    else:
        train_indices, test_indices = train_test_split(
            np.arange(len(labels)),
            test_size=0.20,
            random_state=42,
            stratify=labels,
        )
        split_description = (
            "random validation (collect at least two separate sessions per label "
            "for a more honest test)"
        )
    return train_indices, test_indices, split_description


def rejection_report(model, features, labels, confidence_threshold, margin_threshold):
    probabilities = model.predict_proba(features)
    order = np.argsort(probabilities, axis=1)
    best_indices = order[:, -1]
    second_indices = order[:, -2]
    rows = np.arange(len(features))
    best_scores = probabilities[rows, best_indices]
    margins = best_scores - probabilities[rows, second_indices]
    predictions = model.classes_[best_indices]
    accepted = (
        (best_scores >= confidence_threshold)
        & (margins >= margin_threshold)
        & (predictions != "OTHER")
    )
    if np.any(accepted):
        accepted_accuracy = accuracy_score(labels[accepted], predictions[accepted])
    else:
        accepted_accuracy = 0.0
    coverage = float(np.mean(accepted))
    return accepted_accuracy, coverage


def main() -> None:
    arguments = parse_arguments()
    features, labels, sessions, feature_names = load_dataset(arguments.data)
    counts = Counter(labels)

    if len(counts) < 2:
        raise SystemExit("Collect at least two different gesture labels before training.")
    too_small = {label: count for label, count in counts.items() if count < 20}
    if too_small:
        raise SystemExit(f"Each label needs at least 20 samples. Too small: {too_small}")
    if len(feature_names) not in (HAND_FEATURE_COUNT, HAND_FEATURE_COUNT + POSE_FEATURE_COUNT):
        raise SystemExit(
            f"Unexpected feature count {len(feature_names)}. Expected {HAND_FEATURE_COUNT} "
            f"(hand) or {HAND_FEATURE_COUNT + POSE_FEATURE_COUNT} (hand + arm)."
        )

    train_indices, test_indices, split_description = make_split(features, labels, sessions)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                SVC(
                    C=10.0,
                    kernel="rbf",
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(features[train_indices], labels[train_indices])

    predictions = model.predict(features[test_indices])
    accuracy = accuracy_score(labels[test_indices], predictions)
    print(f"Samples: {len(labels)}")
    print(f"Classes: {dict(sorted(counts.items()))}")
    print(f"Validation method: {split_description}")
    print(f"Validation accuracy: {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(labels[test_indices], predictions, digits=3, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted):")
    print("Labels:", list(model.classes_))
    print(confusion_matrix(labels[test_indices], predictions, labels=model.classes_))

    accepted_accuracy, coverage = rejection_report(
        model,
        features[test_indices],
        labels[test_indices],
        arguments.confidence,
        arguments.margin,
    )
    print(
        f"\nWith rejection rules: {accepted_accuracy * 100:.2f}% accuracy on "
        f"{coverage * 100:.1f}% of validation frames."
    )
    print("Rejected frames become 'Unknown' instead of forcing a possibly wrong label.")

    bundle = {
        "pipeline": model,
        "feature_count": len(feature_names),
        "use_pose": len(feature_names) == HAND_FEATURE_COUNT + POSE_FEATURE_COUNT,
        "confidence_threshold": arguments.confidence,
        "margin_threshold": arguments.margin,
        "smoothing_window": arguments.window,
        "required_votes": arguments.votes,
        "classes": list(model.classes_),
        "dataset": str(arguments.data.resolve()),
    }
    arguments.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, arguments.model)
    print(f"\nSaved model: {arguments.model}")


if __name__ == "__main__":
    main()

