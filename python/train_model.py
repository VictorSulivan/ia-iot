#!/usr/bin/env python3
"""Entraîne un classificateur de gestes à partir de data/raw/*.csv.

Sauvegarde :
  artifacts/gesture_classifier.joblib
  artifacts/gesture_classifier_meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from gesture_dataset import (  # noqa: E402
    LABELS,
    default_raw_dir,
    extract_trial_features,
    feature_matrix,
    load_gesture_csvs,
    repo_root,
)

RANDOM_STATE = 42
DEFAULT_ARTIFACTS = repo_root() / "artifacts"


def build_candidates(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    return {
        "random_forest": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=2.0,
                        gamma="scale",
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def split_by_trial(X, y, groups, random_state: int = RANDOM_STATE):
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_val_idx, test_idx = next(gss_outer.split(X, y, groups))
    X_train_val, y_train_val = X.iloc[train_val_idx], y.iloc[train_val_idx]
    groups_train_val = groups.iloc[train_val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
    train_idx, val_idx = next(gss_inner.split(X_train_val, y_train_val, groups_train_val))
    X_train, y_train = X_train_val.iloc[train_idx], y_train_val.iloc[train_idx]
    X_val, y_val = X_train_val.iloc[val_idx], y_train_val.iloc[val_idx]
    return (X_train, y_train, X_val, y_val, X_train_val, y_train_val, X_test, y_test)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entraîne le classificateur de gestes MPU6050.")
    p.add_argument("--raw-dir", type=Path, default=default_raw_dir())
    p.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    p.add_argument(
        "--model",
        choices=("auto", "random_forest", "svm_rbf", "logreg"),
        default="auto",
        help="Modèle à garder (auto = meilleur F1 sur validation)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Chargement CSV depuis {args.raw_dir}…")
    df = load_gesture_csvs(args.raw_dir)
    features = extract_trial_features(df)
    X, y, groups = feature_matrix(features)
    print(f"Essais : {len(features)} | features : {X.shape[1]}")
    print(y.value_counts().reindex(LABELS).fillna(0).astype(int).to_string())

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_train_val,
        y_train_val,
        X_test,
        y_test,
    ) = split_by_trial(X, y, groups)

    candidates = build_candidates()
    if args.model != "auto":
        candidates = {args.model: candidates[args.model]}

    val_scores: list[tuple[str, float, float]] = []
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)
        acc = accuracy_score(y_val, pred)
        f1 = f1_score(y_val, pred, average="macro", labels=list(LABELS))
        val_scores.append((name, acc, f1))
        print(f"{name:15s}  val_acc={acc:.3f}  val_f1_macro={f1:.3f}")

    best_name = sorted(val_scores, key=lambda t: t[2], reverse=True)[0][0]
    print(f"\nMeilleur modèle (val) : {best_name}")

    final = build_candidates()[best_name]
    final.fit(X_train_val, y_train_val)
    y_pred = final.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="macro", labels=list(LABELS))
    print(f"Test accuracy : {test_acc:.3f}")
    print(f"Test F1 macro : {test_f1:.3f}")
    print()
    print(classification_report(y_test, y_pred, labels=list(LABELS), digits=3))

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifacts_dir / "gesture_classifier.joblib"
    meta_path = args.artifacts_dir / "gesture_classifier_meta.json"

    # probability=True for RF is native; for SVM we enabled it above
    joblib.dump(
        {
            "model": final,
            "feature_columns": list(X.columns),
            "labels": list(LABELS),
            "model_name": best_name,
        },
        model_path,
    )
    meta = {
        "model_name": best_name,
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "n_features": int(X.shape[1]),
        "feature_columns": list(X.columns),
        "labels": list(LABELS),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Modèle sauvé : {model_path}")
    print(f"Meta sauvé   : {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
