# ia-iot — Capture de gestes MPU6050

Outil de collecte de données IMU (Arduino + MPU6050) pour entraîner un classificateur de mouvements :

- `lateral` — mouvement latéral
- `horizontal` — mouvement horizontal
- `circle` — cercle

## Matériel

- Carte Arduino / ESP32 compatible avec `Wire.begin(SDA, SCL)`
- Capteur **MPU6050**
- Broches I2C par défaut du sketch :
  - **SDA → pin 8**
  - **SCL → pin 9**
- Bibliothèque Arduino : **MPU6050** (ex. Electronic Cats `MPU6050`)

## Firmware Arduino

Sketch : [`arduino/mpu6050_capture/mpu6050_capture.ino`](arduino/mpu6050_capture/mpu6050_capture.ino)

1. Ouvrir le sketch dans l’IDE Arduino
2. Installer la bibliothèque `MPU6050`
3. Sélectionner la bonne carte / port
4. Flasher

**Protocole série (115200 baud)** : l’hôte envoie `s` → la carte envoie 200 lignes `ax,ay,az,gx,gy,gz` (1 toutes les 30 ms, ~6 s) → puis `DONE`.

## Capture Python

### Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Lancer

```bash
# Menu interactif (recommandé) — détecte le port si possible
python python/capture_motions.py

# Port explicite
python python/capture_motions.py --port COM3          # Windows
python python/capture_motions.py --port /dev/ttyUSB0  # Linux
python python/capture_motions.py --port /dev/cu.usbserial-XXXX  # macOS

# Un seul essai d’une classe
python python/capture_motions.py --label lateral --once

# Lister les ports
python python/capture_motions.py --list-ports
```

### Déroulement

1. Brancher l’Arduino déjà flashé
2. Lancer le script
3. Choisir le label (`1` / `2` / `3` ou nom)
4. Compte à rebours → effectuer le geste pendant ~6 s
5. Le CSV est écrit ; recommencer ou `q` pour quitter

Objectif typique : **plusieurs essais par classe** (ex. 20–50) pour avoir assez de données d’entraînement.

## Layout CSV (ML-friendly)

Un fichier **par essai** :

```text
data/raw/
  lateral/
    lateral_trial_001.csv
    lateral_trial_002.csv
  horizontal/
    horizontal_trial_001.csv
  circle/
    circle_trial_001.csv
```

### Schéma des colonnes

| Colonne         | Description                                      |
|-----------------|--------------------------------------------------|
| `sample_index`  | 0 … 199                                          |
| `timestamp_ms`  | Temps relatif (`sample_index * 30`)              |
| `ax` … `gz`     | Accéléromètre / gyroscope bruts (int16)          |
| `label`         | `lateral` \| `horizontal` \| `circle`            |
| `trial_id`      | Identifiant d’essai, ex. `lateral_001`           |
| `session`       | Horodatage UTC de la session (`YYYYMMDDTHHMMSSZ`)|

Pas de code d’entraînement dans ce dépôt pour l’instant — capture + export CSV uniquement.
