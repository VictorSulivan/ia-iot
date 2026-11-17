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
            f"{prefix}_mad": np.nan,
            f"{prefix}_zcr": np.nan,
        }
    centered = x - np.mean(x)
    zcr = float(np.sum((centered[:-1] * centered[1:]) < 0)) if x.size > 1 else 0.0
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_ptp": float(np.ptp(x)),
        f"{prefix}_energy": float(np.sum(x.astype(float) ** 2) / len(x)),
        f"{prefix}_abs_mean": float(np.mean(np.abs(x))),
        # mean abs diff ≈ roughness / jerk proxy
        f"{prefix}_mad": float(np.mean(np.abs(np.diff(x)))) if x.size > 1 else 0.0,
        f"{prefix}_zcr": zcr,
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _fft_peak(x: np.ndarray, sample_rate_hz: float = SAMPLE_RATE_HZ) -> tuple[float, float]:
    """Fréquence dominante (Hz) et part d'énergie spectrale du pic (hors DC)."""
    x = np.asarray(x, dtype=float)
    if x.size < 4:
        return 0.0, 0.0
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate_hz)
    if spec.size <= 1:
        return 0.0, 0.0
    # ignorer DC
    i = int(np.argmax(spec[1:])) + 1
    total = float(np.sum(spec[1:]) + 1e-9)
    return float(freqs[i]), float(spec[i] / total)


def _circle_motion_features(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
) -> dict[str, float]:
    """Features utiles pour distinguer un rond d'un balayage horizontal/vertical."""
    gyro_ptp = np.array([np.ptp(gx), np.ptp(gy), np.ptp(gz)], dtype=float)
    accel_ptp = np.array([np.ptp(ax), np.ptp(ay), np.ptp(az)], dtype=float)
    order = np.argsort(gyro_ptp)[::-1]
    top2 = float(gyro_ptp[order[0]] + gyro_ptp[order[1]])
    gyro_sum = float(np.sum(gyro_ptp) + 1e-9)
    planar_gyro_ratio = top2 / gyro_sum
    # isotropie : un cercle mobilise 2 axes ; un geste linéaire en domine 1
    gyro_axis_entropy = float(
        -np.sum((gyro_ptp / gyro_sum) * np.log((gyro_ptp / gyro_sum) + 1e-12))
    )
    accel_sum = float(np.sum(accel_ptp) + 1e-9)
    planar_accel_ratio = float(
        (accel_ptp[np.argsort(accel_ptp)[::-1][0]] + accel_ptp[np.argsort(accel_ptp)[::-1][1]])
        / accel_sum
    )

    feats: dict[str, float] = {
        "corr_ax_ay": _corr(ax, ay),
        "corr_ax_az": _corr(ax, az),
        "corr_ay_az": _corr(ay, az),
        "corr_gx_gy": _corr(gx, gy),
        "corr_gx_gz": _corr(gx, gz),
        "corr_gy_gz": _corr(gy, gz),
        "planar_gyro_ratio": planar_gyro_ratio,
        "planar_accel_ratio": planar_accel_ratio,
        "gyro_axis_entropy": gyro_axis_entropy,
        "gyro_ptp_min": float(np.min(gyro_ptp)),
        "gyro_ptp_ratio_12": float(gyro_ptp[order[0]] / (gyro_ptp[order[1]] + 1e-9)),
    }
    for name, sig in (("ax", ax), ("ay", ay), ("gx", gx), ("gy", gy), ("gz", gz)):
        hz, pow_frac = _fft_peak(sig)
        feats[f"fft_{name}_hz"] = hz
        feats[f"fft_{name}_pow"] = pow_frac
    return feats


def extract_trial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features statistiques + temporelles/spectrales au niveau essai."""
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
        accel_ptp = [np.ptp(ax), np.ptp(ay), np.ptp(az)]
        gyro_ptp = [np.ptp(gx), np.ptp(gy), np.ptp(gz)]
        feats["accel_dom_axis"] = int(np.argmax(accel_ptp))
        feats["gyro_dom_axis"] = int(np.argmax(gyro_ptp))
        feats.update(_circle_motion_features(ax, ay, az, gx, gy, gz))
        feature_rows.append(feats)

    return pd.DataFrame(feature_rows)


def augment_trial_dataframe(
    df: pd.DataFrame,
    *,
    n_aug_per_trial: int = 2,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Augmente les essais (bruit + scale) pour réduire le décalage live / train.

    Utile surtout pour les cercles, souvent plus lents/petits en live.
    """
    rng = rng or np.random.default_rng(42)
    frames = [df]
    for (label, trial_id), g in df.groupby(["label", "trial_id"], sort=True):
        g = g.sort_values("sample_index")
        for k in range(n_aug_per_trial):
            # cercles : plus d'augmentations "faibles" ; autres : léger bruit
            if label == "circle":
                scale = float(rng.uniform(0.55, 1.15))
                noise_sigma = float(rng.uniform(20.0, 120.0))
            else:
                scale = float(rng.uniform(0.85, 1.15))
                noise_sigma = float(rng.uniform(10.0, 60.0))
            aug = g.copy()
            for col in SENSOR_COLS:
                base = aug[col].to_numpy(dtype=float)
                # conserver un offset proche de la gravité sur az
                if col == "az":
                    centered = base - np.mean(base)
                    aug[col] = (np.mean(base) + centered * scale) + rng.normal(
                        0.0, noise_sigma, size=len(base)
                    )
                else:
                    aug[col] = base * scale + rng.normal(0.0, noise_sigma, size=len(base))
            aug["trial_id"] = f"{trial_id}_aug{k+1}"
            frames.append(aug)
    return pd.concat(frames, ignore_index=True)


def feature_matrix(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """X, y, groups (trial_id) pour split sans fuite."""
    meta = {"label", "trial_id"}
    x_cols = [c for c in features.columns if c not in meta]
    X = features[x_cols].copy()
    y = features["label"].copy()
    groups = features["trial_id"].copy()
    return X, y, groups


def samples_to_dataframe(
    rows: list[dict],
    *,
    label: str = "unknown",
    trial_id: str = "live_001",
) -> pd.DataFrame:
    """Convertit une liste d'échantillons (dict ax..gz) en DataFrame compatible features."""
    if not rows:
        raise ValueError("Aucun échantillon fourni")
    records = []
    for i, row in enumerate(rows):
        records.append(
            {
                "sample_index": int(row.get("sample_index", i)),
                "timestamp_ms": int(row.get("timestamp_ms", i * int(1000 / SAMPLE_RATE_HZ))),
                "ax": int(row["ax"]),
                "ay": int(row["ay"]),
                "az": int(row["az"]),
                "gx": int(row["gx"]),
                "gy": int(row["gy"]),
                "gz": int(row["gz"]),
                "label": label,
                "trial_id": trial_id,
                "session": row.get("session", "live"),
            }
        )
    return pd.DataFrame.from_records(records)


def feature_vector_from_samples(
    rows: list[dict],
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Une ligne de features pour un essai live (colonnes alignées si fournies)."""
    df = samples_to_dataframe(rows)
    feats = extract_trial_features(df)
    meta = {"label", "trial_id"}
    X = feats.drop(columns=[c for c in meta if c in feats.columns])
    if feature_columns is not None:
        missing = [c for c in feature_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Features manquantes pour l'inférence: {missing}")
        X = X[feature_columns]
    return X
