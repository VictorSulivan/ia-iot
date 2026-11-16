#!/usr/bin/env python3
"""Capture MPU6050 gesture trials over serial for ML training.

Protocol: send 's' -> receive 75 lines of ax,ay,az,gx,gy,gz (50 Hz, 1.5 s) -> DONE.

Layout: one CSV per gesture label (append trials):
  data/raw/vertical.csv
  data/raw/horizontal.csv
  data/raw/circle.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LABELS = ("vertical", "horizontal", "circle")
SAMPLE_RATE_HZ = 50
CAPTURE_DURATION_S = 1.5
NUM_SAMPLES = int(SAMPLE_RATE_HZ * CAPTURE_DURATION_S)  # 75
INTERVAL_MS = int(1000 / SAMPLE_RATE_HZ)  # 20
BAUD = 115200
DONE_TOKEN = "DONE"
DEFAULT_TARGET_TRIALS = 200
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


def label_csv_path(output_dir: Path, label: str) -> Path:
    return output_dir / f"{label}.csv"


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

    # Consommer DONE s'il arrive après les N lignes
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


def count_trials_in_csv(path: Path) -> int:
    """Nombre d'essais distincts déjà présents dans le CSV du label."""
    if not path.exists() or path.stat().st_size == 0:
        return 0
    trial_ids: set[str] = set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "trial_id" not in reader.fieldnames:
            return 0
        for row in reader:
            tid = (row.get("trial_id") or "").strip()
            if tid:
                trial_ids.add(tid)
    return len(trial_ids)


def next_trial_number(path: Path, label: str) -> int:
    """Prochain numéro d'essai (1-based), basé sur les trial_id existants."""
    if not path.exists() or path.stat().st_size == 0:
        return 1
    max_n = 0
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "trial_id" not in reader.fieldnames:
            return 1
        for row in reader:
            tid = (row.get("trial_id") or "").strip()
            # Formats acceptés: "001", "vertical_001"
            suffix = tid.rsplit("_", 1)[-1] if tid else ""
            try:
                max_n = max(max_n, int(suffix))
            except ValueError:
                continue
    return max_n + 1


def append_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def prompt_label(default: str | None = None) -> str | None:
    print()
    print("Labels: 1=vertical  2=horizontal  3=circle  q=quitter")
    if default:
        prompt = f"Choix [{default}]: "
    else:
        prompt = "Choix: "
    choice = input(prompt).strip().lower()
    if choice in ("q", "quit", "exit"):
        return None
    if not choice and default:
        return default
    mapping = {"1": "vertical", "2": "horizontal", "3": "circle"}
    if choice in mapping:
        return mapping[choice]
    if choice in LABELS:
        return choice
    print(f"Choix invalide: {choice!r}")
    return prompt_label(default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture de gestes MPU6050 vers un CSV par label "
            "(vertical.csv / horizontal.csv / circle.csv)."
        )
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
        help="Répertoire des CSV (défaut: data/raw/)",
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=3,
        help="Compte à rebours avant capture (secondes, défaut: 3)",
    )
    parser.add_argument(
        "--target-trials",
        type=int,
        default=DEFAULT_TARGET_TRIALS,
        help=(
            f"Objectif d'essais par label (défaut: {DEFAULT_TARGET_TRIALS}). "
            "Affiché en progression ; 0 = pas d'objectif."
        ),
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


def print_label_progress(output_dir: Path, target: int) -> None:
    print("Progression par label:")
    for label in LABELS:
        path = label_csv_path(output_dir, label)
        n = count_trials_in_csv(path)
        if target > 0:
            print(f"  {label:11s} {n}/{target} essais → {path.name}")
        else:
            print(f"  {label:11s} {n} essais → {path.name}")


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
    output_dir.mkdir(parents=True, exist_ok=True)
    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = max(0, args.target_trials)

    serial, _ = _require_serial()
    print(f"Connexion à {port} @ {args.baud}…")
    try:
        ser = open_serial(port, args.baud, timeout=1.0)
    except serial.SerialException as exc:
        print(f"Impossible d'ouvrir {port}: {exc}", file=sys.stderr)
        return 1

    print(f"Session: {session}")
    print(f"Sortie:  {output_dir.resolve()}")
    print("Layout:  data/raw/<label>.csv  (1 fichier par geste, essais ajoutés en append)")
    print(
        f"Chaque essai = {NUM_SAMPLES} échantillons "
        f"({SAMPLE_RATE_HZ} Hz, {CAPTURE_DURATION_S} s)."
    )
    if target > 0:
        print(f"Objectif: {target} essais (mouvements) par geste.")
    print_label_progress(output_dir, target)

    last_label = args.label
    captures = 0

    try:
        while True:
            label = args.label if (args.once and args.label) else prompt_label(last_label)
            if label is None:
                break
            last_label = label

            out_path = label_csv_path(output_dir, label)
            done = count_trials_in_csv(out_path)
            if target > 0 and done >= target:
                print(
                    f"{label}: objectif déjà atteint ({done}/{target}). "
                    "Choisissez un autre label ou q."
                )
                if args.once:
                    break
                continue

            trial_num = next_trial_number(out_path, label)
            trial_id = f"{label}_{trial_num:03d}"
            progress = f"{done + 1}/{target}" if target > 0 else f"#{trial_num}"

            print()
            print(f">>> Capture {label} | essai {progress} | → {out_path.name}")
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

            append_csv(out_path, rows)
            captures += 1
            total = count_trials_in_csv(out_path)
            print(
                f"OK — {len(rows)} échantillons ajoutés → {out_path} "
                f"({total}" + (f"/{target}" if target > 0 else "") + " essais)"
            )

            if target > 0 and total >= target:
                print(f"Objectif atteint pour {label} ({total}/{target}).")

            if args.once:
                break
    except KeyboardInterrupt:
        print("\nInterrompu.")
    finally:
        ser.close()

    print()
    print_label_progress(output_dir, target)
    print(f"Terminé. {captures} essai(s) ajouté(s) cette session.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
