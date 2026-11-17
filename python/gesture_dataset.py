"""Chargement et features des CSV de gestes MPU6050.

Format attendu (un fichier par label sous data/raw/) :
  vertical.csv | horizontal.csv | circle.csv
Colonnes :
  sample_index,timestamp_ms,ax,ay,az,gx,gy,gz,label,trial_id,session

Chaque essai = 75 échantillons (50 Hz × 1,5 s).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ("vertical", "horizontal", "circle")
SENSOR_COLS = ("ax", "ay", "az", "gx", "gy", "gz")
EXPECTED_SAMPLES_PER_TRIAL = 75
SAMPLE_RATE_HZ = 50
CAPTURE_DURATION_S = 1.5


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_raw_dir() -> Path:
    return repo_root() / "data" / "raw"


def load_gesture_csvs(raw_dir: Path | None = None) -> pd.DataFrame:
    """Charge et concatène les CSV labelés. Lève FileNotFoundError si aucun fichier."""
    raw_dir = Path(raw_dir) if raw_dir is not None else default_raw_dir()
    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for label in LABELS:
        path = raw_dir / f"{label}.csv"
        if not path.exists() or path.stat().st_size == 0:
            missing.append(str(path))
            continue
        df = pd.read_csv(path)
        required = {
            "sample_index",
            "timestamp_ms",
            *SENSOR_COLS,
            "label",
            "trial_id",
            "session",
        }
        absent = required - set(df.columns)
        if absent:
            raise ValueError(f"{path}: colonnes manquantes {sorted(absent)}")
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "Aucun CSV de geste trouvé. Placez "
            + ", ".join(f"{l}.csv" for l in LABELS)
            + f" dans {raw_dir} "
            "(produits par python/capture_motions.py)."
        )

    data = pd.concat(frames, ignore_index=True)
    for col in SENSOR_COLS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["sample_index"] = pd.to_numeric(data["sample_index"], errors="coerce")
    data["timestamp_ms"] = pd.to_numeric(data["timestamp_ms"], errors="coerce")
    data["label"] = data["label"].astype(str).str.strip()
    data["trial_id"] = data["trial_id"].astype(str).str.strip()
    return data


def trial_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Résumé par essai : taille, durée, fréq. estimée."""
    rows = []
    for (label, trial_id), g in df.groupby(["label", "trial_id"], sort=True):
        g = g.sort_values("sample_index")
        n = len(g)
        t0 = g["timestamp_ms"].iloc[0]
        t1 = g["timestamp_ms"].iloc[-1]
        duration_ms = float(t1 - t0) if n > 1 else 0.0
        # n échantillons → n-1 intervalles
        hz = ((n - 1) / (duration_ms / 1000.0)) if duration_ms > 0 else np.nan
        rows.append(
            {
                "label": label,
                "trial_id": trial_id,
                "n_samples": n,
                "duration_ms": duration_ms,
                "sample_rate_hz": hz,
                "session": g["session"].iloc[0] if "session" in g.columns else None,
            }
        )
    return pd.DataFrame(rows)


def _stats_for_series(x: np.ndarray, prefix: str) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_ptp": np.nan,
            f"{prefix}_energy": np.nan,
            f"{prefix}_abs_mean": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_ptp": float(np.ptp(x)),
        f"{prefix}_energy": float(np.sum(x.astype(float) ** 2) / len(x)),
        f"{prefix}_abs_mean": float(np.mean(np.abs(x))),
    }


def extract_trial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features statistiques au niveau essai (pas de fuite inter-échantillons)."""
    feature_rows: list[dict] = []
    for (label, trial_id), g in df.groupby(["label", "trial_id"], sort=True):
        g = g.sort_values("sample_index")
        feats: dict = {"label": label, "trial_id": trial_id}
        for col in SENSOR_COLS:
            feats.update(_stats_for_series(g[col].to_numpy(), col))

        ax, ay, az = g["ax"].to_numpy(float), g["ay"].to_numpy(float), g["az"].to_numpy(float)
        gx, gy, gz = g["gx"].to_numpy(float), g["gy"].to_numpy(float), g["gz"].to_numpy(float)
        amag = np.sqrt(ax**2 + ay**2 + az**2)
        gmag = np.sqrt(gx**2 + gy**2 + gz**2)
        feats.update(_stats_for_series(amag, "a_mag"))
        feats.update(_stats_for_series(gmag, "g_mag"))
        # Dominance d'axe (indice du canal avec le plus grand peak-to-peak)
        accel_ptp = [np.ptp(ax), np.ptp(ay), np.ptp(az)]
        gyro_ptp = [np.ptp(gx), np.ptp(gy), np.ptp(gz)]
        feats["accel_dom_axis"] = int(np.argmax(accel_ptp))
        feats["gyro_dom_axis"] = int(np.argmax(gyro_ptp))
        feature_rows.append(feats)

    return pd.DataFrame(feature_rows)


def feature_matrix(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """X, y, groups (trial_id) pour split sans fuite."""
    meta = {"label", "trial_id"}
    x_cols = [c for c in features.columns if c not in meta]
    X = features[x_cols].copy()
    y = features["label"].copy()
    groups = features["trial_id"].copy()
    return X, y, groups
