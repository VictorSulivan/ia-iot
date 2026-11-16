#!/usr/bin/env python3
"""Capture MPU6050 gesture trials over serial for ML training.

Protocol: send 's' -> receive 200 lines of ax,ay,az,gx,gy,gz -> DONE.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LABELS = ("lateral", "horizontal", "circle")
NUM_SAMPLES = 200
INTERVAL_MS = 30
BAUD = 115200
DONE_TOKEN = "DONE"
CSV_COLUMNS = (
    "sample_index",
    "timestamp_ms",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "label",
    "trial_id",
    "session",
)


def _require_serial():
    try:
        import serial
        from serial.tools import list_ports
    except ImportError:
        print(
            "pyserial manquant. Installez avec: pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return serial, list_ports


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "raw"


def list_serial_ports() -> list[str]:
    _, list_ports = _require_serial()
    return [p.device for p in list_ports.comports()]


def autodetect_port() -> str | None:
    ports = list_serial_ports()
    if not ports:
        return None
    preferred = []
    for device in ports:
        lower = device.lower()
        if any(token in lower for token in ("usb", "acm", "tty.usb", "cu.usb", "com")):
            preferred.append(device)
    return preferred[0] if preferred else ports[0]


def resolve_port(cli_port: str | None) -> str:
    if cli_port:
        return cli_port
    detected = autodetect_port()
    if detected:
        print(f"Port auto-détecté: {detected}")
        return detected
    available = list_serial_ports()
    if available:
        print("Ports série disponibles:")
        for i, device in enumerate(available, start=1):
            print(f"  {i}. {device}")
        choice = input("Choisissez un numéro (ou tapez le chemin du port): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(available):
            return available[int(choice) - 1]
        if choice:
            return choice
    raise SystemExit(
        "Aucun port série trouvé. Branchez l'Arduino et/ou passez --port COMx / /dev/ttyUSB0"
    )


def countdown(seconds: int) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"  Départ dans {remaining}…", flush=True)
        time.sleep(1)
    print("  GO ! Effectuez le geste maintenant.", flush=True)


def open_serial(port: str, baud: int, timeout: float):
    serial, _ = _require_serial()
    ser = serial.Serial(port=port, baudrate=baud, timeout=timeout)
    # Laisser le temps au reset USB / bootloader
    time.sleep(2.0)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    return ser


def capture_trial(ser, label: str, trial_id: str, session: str) -> list[dict]:
    ser.reset_input_buffer()
    ser.write(b"s")
    ser.flush()

    rows: list[dict] = []
    deadline = time.monotonic() + (NUM_SAMPLES * INTERVAL_MS / 1000.0) + 10.0

    while len(rows) < NUM_SAMPLES:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timeout: {len(rows)}/{NUM_SAMPLES} échantillons reçus avant DONE"
            )
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        if line == DONE_TOKEN:
            break
        parts = line.split(",")
        if len(parts) != 6:
            print(f"  Ligne ignorée (format inattendu): {line!r}")
            continue
        try:
            ax, ay, az, gx, gy, gz = (int(p.strip()) for p in parts)
        except ValueError:
            print(f"  Ligne ignorée (valeurs non entières): {line!r}")
            continue
        idx = len(rows)
        rows.append(
            {
                "sample_index": idx,
                "timestamp_ms": idx * INTERVAL_MS,
                "ax": ax,
                "ay": ay,
                "az": az,
                "gx": gx,
                "gy": gy,
                "gz": gz,
                "label": label,
                "trial_id": trial_id,
                "session": session,
            }
        )

    # Consommer DONE s'il arrive après les 200 lignes
    if len(rows) == NUM_SAMPLES:
        leftover_deadline = time.monotonic() + 2.0
        while time.monotonic() < leftover_deadline:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if line == DONE_TOKEN:
                break

    if len(rows) != NUM_SAMPLES:
        raise RuntimeError(
            f"Capture incomplète: {len(rows)}/{NUM_SAMPLES} échantillons pour {label}/{trial_id}"
        )
    return rows


def next_trial_id(label_dir: Path, label: str) -> str:
    existing = sorted(label_dir.glob(f"{label}_trial_*.csv"))
    max_n = 0
    for path in existing:
        stem = path.stem  # lateral_trial_003
        try:
            n = int(stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        max_n = max(max_n, n)
    return f"{max_n + 1:03d}"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def prompt_label(default: str | None = None) -> str | None:
    print()
    print("Labels: 1=lateral  2=horizontal  3=circle  q=quitter")
    if default:
        prompt = f"Choix [{default}]: "
    else:
        prompt = "Choix: "
    choice = input(prompt).strip().lower()
    if choice in ("q", "quit", "exit"):
        return None
    if not choice and default:
        return default
    mapping = {"1": "lateral", "2": "horizontal", "3": "circle"}
    if choice in mapping:
        return mapping[choice]
    if choice in LABELS:
        return choice
    print(f"Choix invalide: {choice!r}")
    return prompt_label(default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture de gestes MPU6050 (lateral / horizontal / circle) vers CSV."
    )
    parser.add_argument(
        "--port",
        help="Port série (ex: COM3, /dev/ttyUSB0, /dev/cu.usbserial-*). Auto-détecté si omis.",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=BAUD,
        help=f"Baud rate (défaut: {BAUD})",
    )
    parser.add_argument(
        "--label",
        choices=LABELS,
        help="Label du geste. Si omis, menu interactif.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Répertoire racine des CSV (défaut: data/raw/)",
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=3,
        help="Compte à rebours avant capture (secondes, défaut: 3)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Une seule capture puis sortie (sinon boucle jusqu'à q)",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="Lister les ports série et quitter",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_ports:
        ports = list_serial_ports()
        if not ports:
            print("Aucun port série détecté.")
            return 1
        for device in ports:
            print(device)
        return 0

    port = resolve_port(args.port)
    output_dir: Path = args.output_dir
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    serial, _ = _require_serial()
    print(f"Connexion à {port} @ {args.baud}…")
    try:
        ser = open_serial(port, args.baud, timeout=1.0)
    except serial.SerialException as exc:
        print(f"Impossible d'ouvrir {port}: {exc}", file=sys.stderr)
        return 1

    print(f"Session: {session}")
    print(f"Sortie:  {output_dir.resolve()}")
    print("Layout:  data/raw/<label>/<label>_trial_NNN.csv  (1 fichier = 1 essai)")

    last_label = args.label
    captures = 0

    try:
        while True:
            label = args.label if (args.once and args.label) else prompt_label(last_label)
            if label is None:
                break
            last_label = label

            label_dir = output_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            trial_num = next_trial_id(label_dir, label)
            trial_id = f"{label}_{trial_num}"
            out_path = label_dir / f"{label}_trial_{trial_num}.csv"

            print()
            print(f">>> Capture {label} | trial {trial_num} | fichier {out_path.name}")
            print("Préparez-vous à effectuer le geste.")
            countdown(max(0, args.countdown))

            try:
                rows = capture_trial(ser, label=label, trial_id=trial_id, session=session)
            except (TimeoutError, RuntimeError) as exc:
                print(f"Échec: {exc}", file=sys.stderr)
                if args.once:
                    return 1
                retry = input("Réessayer ? [O/n]: ").strip().lower()
                if retry in ("n", "non", "no"):
                    continue
                continue

            write_csv(out_path, rows)
            captures += 1
            print(f"OK — {len(rows)} échantillons → {out_path}")

            if args.once:
                break
    except KeyboardInterrupt:
        print("\nInterrompu.")
    finally:
        ser.close()

    print(f"Terminé. {captures} essai(s) enregistré(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
