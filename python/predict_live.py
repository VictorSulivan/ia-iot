#!/usr/bin/env python3
"""Classification live des gestes MPU6050 (vertical / horizontal / circle=rond).

1. Charge artifacts/gesture_classifier.joblib (entraîné via train_model.py)
2. Envoie 's' à l'Arduino → 75 échantillons (50 Hz, 1,5 s)
3. Affiche le geste prédit (+ probabilités si disponibles)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from capture_motions import (  # noqa: E402
    BAUD,
    capture_trial,
    countdown,
    open_serial,
    resolve_port,
)
from gesture_dataset import (  # noqa: E402
    CAPTURE_DURATION_S,
    EXPECTED_SAMPLES_PER_TRIAL,
    SAMPLE_RATE_HZ,
    feature_vector_from_samples,
    repo_root,
)

DISPLAY_NAMES = {
    "vertical": "VERTICAL",
    "horizontal": "HORIZONTAL",
    "circle": "ROND (circle)",
}

DEFAULT_MODEL = repo_root() / "artifacts" / "gesture_classifier.joblib"


def load_artifact(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"Modèle introuvable : {path}\n"
            "Entraînez d'abord avec : python python/train_model.py"
        )
    data = joblib.load(path)
    required = {"model", "feature_columns", "labels"}
    missing = required - set(data)
    if missing:
        raise SystemExit(f"Artefact invalide ({path}), clés manquantes : {sorted(missing)}")
    return data


def predict_rows(artifact: dict, rows: list[dict]) -> tuple[str, dict[str, float] | None]:
    model = artifact["model"]
    feature_columns = list(artifact["feature_columns"])
    X = feature_vector_from_samples(rows, feature_columns=feature_columns)
    pred = model.predict(X)[0]
    probs = None
    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(X)[0]
            classes = list(getattr(model, "classes_", artifact["labels"]))
            # Pipeline expose classes_ via le step final
            if hasattr(model, "named_steps") and "clf" in model.named_steps:
                classes = list(model.named_steps["clf"].classes_)
            probs = {str(c): float(v) for c, v in zip(classes, p)}
        except Exception:
            probs = None
    return str(pred), probs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prédit le geste (vertical / horizontal / rond) depuis le MPU6050."
    )
    p.add_argument("--port", help="Port série (COM3, /dev/ttyUSB0, …)")
    p.add_argument("--baud", type=int, default=BAUD)
    p.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Chemin joblib (défaut: {DEFAULT_MODEL})",
    )
    p.add_argument("--countdown", type=int, default=3)
    p.add_argument(
        "--once",
        action="store_true",
        help="Une seule prédiction puis sortie",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    artifact = load_artifact(args.model)
    print(f"Modèle : {args.model} ({artifact.get('model_name', '?')})")
    print(f"Classes : {artifact['labels']}")
    print(
        f"Capture : {EXPECTED_SAMPLES_PER_TRIAL} samples @ {SAMPLE_RATE_HZ} Hz "
        f"({CAPTURE_DURATION_S} s)"
    )

    port = resolve_port(args.port)
    print(f"Connexion {port} @ {args.baud}…")
    try:
        ser = open_serial(port, args.baud, timeout=1.0)
    except Exception as exc:  # serial.SerialException
        print(f"Impossible d'ouvrir {port}: {exc}", file=sys.stderr)
        return 1

    n = 0
    try:
        while True:
            print()
            input("Entrée pour capturer un geste (Ctrl+C pour quitter)… ")
            print("Préparez-vous…")
            countdown(max(0, args.countdown))
            try:
                rows = capture_trial(
                    ser,
                    label="unknown",
                    trial_id=f"live_{n + 1:03d}",
                    session="live",
                )
            except (TimeoutError, RuntimeError) as exc:
                print(f"Échec capture : {exc}", file=sys.stderr)
                if args.once:
                    return 1
                continue

            pred, probs = predict_rows(artifact, rows)
            n += 1
            nice = DISPLAY_NAMES.get(pred, pred.upper())
            print()
            print("=" * 40)
            print(f"  Geste détecté : {nice}")
            print("=" * 40)
            if probs:
                ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
                for label, p in ordered:
                    bar = "█" * int(round(p * 20))
                    print(f"  {DISPLAY_NAMES.get(label, label):16s} {p:5.1%}  {bar}")

            if args.once:
                break
    except KeyboardInterrupt:
        print("\nArrêt.")
    finally:
        ser.close()

    print(f"{n} prédiction(s) effectuée(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
