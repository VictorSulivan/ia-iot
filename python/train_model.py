#!/usr/bin/env python3
"""Entraîne un classificateur de gestes à partir de data/raw/*.csv.

Sauvegarde :
  artifacts/gesture_classifier.joblib
  artifacts/gesture_classifier_meta.json

Améliorations cercle :
  - features temporelles / FFT / corrélations multi-axes
  - augmentation (scale + bruit), plus forte sur les essais circle
  - poids de classe un peu plus élevé pour circle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from gesture_dataset import (  # noqa: E402
    LABELS,
    augment_trial_dataframe,
    default_raw_dir,
    extract_trial_features,
    feature_matrix,
    load_gesture_csvs,
    repo_root,
)

RANDOM_STATE = 42
DEFAULT_ARTIFACTS = repo_root() / "artifacts"

# Favorise un peu le rappel sur circle (souvent confondu avec horizontal en live)
CLASS_WEIGHT = {"vertical": 1.0, "horizontal": 1.0, "circle": 1.8}


def build_candidates(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    return {
        "random_forest": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=1,
                        max_features="sqrt",
                        class_weight=CLASS_WEIGHT,
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
                        C=3.0,
                        gamma="scale",
                        class_weight=CLASS_WEIGHT,
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
                        max_iter=3000,
                        class_weight=CLASS_WEIGHT,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def split_trial_ids(
    features: pd.DataFrame, random_state: int = RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retourne des tableaux de trial_id : train, val, test."""
    X_dummy = features[["trial_id"]]
    y = features["label"]
    groups = features["trial_id"]

    gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_val_idx, test_idx = next(gss_outer.split(X_dummy, y, groups))
    train_val_ids = groups.iloc[train_val_idx]
    y_train_val = y.iloc[train_val_idx]
    test_ids = groups.iloc[test_idx].to_numpy()

    gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
    tr_idx, va_idx = next(gss_inner.split(train_val_ids.to_frame(), y_train_val, train_val_ids))
    train_ids = train_val_ids.iloc[tr_idx].to_numpy()
    val_ids = train_val_ids.iloc[va_idx].to_numpy()
    return train_ids, val_ids, test_ids


def features_for_ids(raw_df: pd.DataFrame, trial_ids: np.ndarray, *, augment: bool) -> pd.DataFrame:
    subset = raw_df[raw_df["trial_id"].isin(set(trial_ids))].copy()
    if augment:
        subset = augment_trial_dataframe(subset, n_aug_per_trial=2, rng=np.random.default_rng(RANDOM_STATE))
    return extract_trial_features(subset)


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
    p.add_argument(
        "--no-augment",
        action="store_true",
        help="Désactive l'augmentation d'essais (scale/bruit)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Chargement CSV depuis {args.raw_dir}…")
    raw = load_gesture_csvs(args.raw_dir)
    base_features = extract_trial_features(raw)
    print(
        f"Essais bruts : {len(base_features)} | "
        f"features/essai : {feature_matrix(base_features)[0].shape[1]}"
    )
    print(base_features["label"].value_counts().reindex(LABELS).fillna(0).astype(int).to_string())

    train_ids, val_ids, test_ids = split_trial_ids(base_features)
    use_aug = not args.no_augment
    print(f"Augmentation train : {'oui' if use_aug else 'non'}")

    feat_train = features_for_ids(raw, train_ids, augment=use_aug)
    feat_val = features_for_ids(raw, val_ids, augment=False)
    feat_test = features_for_ids(raw, test_ids, augment=False)
    feat_train_val = features_for_ids(
        raw, np.concatenate([train_ids, val_ids]), augment=use_aug
    )

    X_train, y_train, _ = feature_matrix(feat_train)
    X_val, y_val, _ = feature_matrix(feat_val)
    X_test, y_test, _ = feature_matrix(feat_test)
    X_train_val, y_train_val, _ = feature_matrix(feat_train_val)

    # Aligner les colonnes (sécurité)
    cols = list(X_train.columns)
    X_val = X_val.reindex(columns=cols)
    X_test = X_test.reindex(columns=cols)
    X_train_val = X_train_val.reindex(columns=cols)

    print(f"Train (aug) : {len(y_train)} | val : {len(y_val)} | test : {len(y_test)}")

    candidates = build_candidates()
    if args.model != "auto":
        candidates = {args.model: candidates[args.model]}

    val_scores: list[tuple[str, float, float, float]] = []
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)
        acc = accuracy_score(y_val, pred)
        f1 = f1_score(y_val, pred, average="macro", labels=list(LABELS))
        recall_c = recall_score(
            y_val, pred, labels=["circle"], average="macro", zero_division=0
        )
        val_scores.append((name, acc, f1, recall_c))
        print(
            f"{name:15s}  val_acc={acc:.3f}  val_f1_macro={f1:.3f}  "
            f"val_recall_circle={recall_c:.3f}"
        )

    # Priorité : F1 macro, puis rappel circle.
    # Pour le live (predict_proba), on préfère RandomForest s'il est quasi aussi bon.
    ranked = sorted(val_scores, key=lambda t: (t[2], t[3]), reverse=True)
    best_name = ranked[0][0]
    rf = next((t for t in val_scores if t[0] == "random_forest"), None)
    if (
        best_name != "random_forest"
        and rf is not None
        and (ranked[0][2] - rf[2]) <= 0.02
        and "random_forest" in candidates
    ):
        print(
            f"Meilleur brut : {best_name} (F1={ranked[0][2]:.3f}) — "
            f"export live = random_forest (F1={rf[2]:.3f}, predict_proba)"
        )
        best_name = "random_forest"
    else:
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
    print("Matrice de confusion (V/H/C) :")
    print(confusion_matrix(y_test, y_pred, labels=list(LABELS)))

    # Robustesse cercle "plus faible" (proxy live)
    circle_test_ids = feat_test.loc[feat_test["label"] == "circle", "trial_id"].unique()
    if len(circle_test_ids):
        weak = raw[raw["trial_id"].isin(set(circle_test_ids))].copy()
        for col in ("ax", "ay", "gx", "gy", "gz"):
            weak[col] = (weak[col].astype(float) * 0.65).round().astype(np.int64)
        # az : scale autour de la moyenne (gravité)
        parts = []
        for _, g in weak.groupby("trial_id", sort=False):
            g = g.copy()
            az = g["az"].astype(float)
            g["az"] = (az.mean() + (az - az.mean()) * 0.65).round().astype(np.int64)
            parts.append(g)
        weak = pd.concat(parts, ignore_index=True)
        weak_feat = extract_trial_features(weak)
        Xw, _, _ = feature_matrix(weak_feat)
        Xw = Xw.reindex(columns=cols)
        weak_pred = final.predict(Xw)
        weak_recall = float(np.mean(weak_pred == "circle"))
        print(f"\nRappel circle sur signaux atténués (×0.65) : {weak_recall:.3f}")

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifacts_dir / "gesture_classifier.joblib"
    meta_path = args.artifacts_dir / "gesture_classifier_meta.json"

    joblib.dump(
        {
            "model": final,
            "feature_columns": cols,
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
        "n_features": int(len(cols)),
        "feature_columns": cols,
        "labels": list(LABELS),
        "class_weight": CLASS_WEIGHT,
        "augment": use_aug,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Modèle sauvé : {model_path}")
    print(f"Meta sauvé   : {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
