# ia-iot — Capture de gestes MPU6050

Outil de collecte de données IMU (Arduino + MPU6050) pour entraîner un classificateur de mouvements :

- `vertical` — mouvement vertical
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

**Protocole série (115200 baud)** : l’hôte envoie `s` → la carte envoie **75** lignes `ax,ay,az,gx,gy,gz` à **50 Hz** (1 toutes les 20 ms, **1,5 s**) → puis `DONE`.

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
python python/capture_motions.py --label vertical --once

# Objectif d’essais par geste (défaut: 200)
python python/capture_motions.py --target-trials 200

# Lister les ports
python python/capture_motions.py --list-ports
```

### Déroulement

1. Brancher l’Arduino déjà flashé
2. Lancer le script
3. Choisir le label (`1` / `2` / `3` ou nom)
4. Compte à rebours → effectuer le geste pendant **1,5 s**
5. Les **75** échantillons (50 Hz) sont **ajoutés** au CSV du label ; recommencer jusqu’à 200 essais, puis `q`

Objectif : **200 mouvements (essais) par geste**, chacun = **75** échantillons IMU (50 Hz × 1,5 s).

## Layout CSV

**Un fichier par geste** (tous les essais du même label sont appendés dedans) :

```text
data/raw/
  vertical.csv      # jusqu’à 200 essais × 75 lignes
  horizontal.csv
  circle.csv
```

Chaque essai ajoute 75 lignes ; la colonne `trial_id` (ex. `vertical_001`) sépare les mouvements.

### Schéma des colonnes

| Colonne         | Description                                      |
|-----------------|--------------------------------------------------|
| `sample_index`  | 0 … 74 dans l’essai                              |
| `timestamp_ms`  | Temps relatif (`sample_index * 20`)              |
| `ax` … `gz`     | Accéléromètre / gyroscope bruts (int16)          |
| `label`         | `vertical` \| `horizontal` \| `circle`            |
| `trial_id`      | Identifiant d’essai, ex. `vertical_001`           |
| `session`       | Horodatage UTC de la session (`YYYYMMDDTHHMMSSZ`)|

Les CSV d’entraînement **doivent être commités** dans `data/raw/` (ils ne sont plus ignorés par git). Si vous avez déjà capturé localement, copiez simplement :

```bash
# depuis la machine de capture
cp vertical.csv horizontal.csv circle.csv /chemin/vers/ia-iot/data/raw/
git add data/raw/*.csv && git commit -m "Add gesture training CSVs" && git push
```

## Reconnaissance live (objectif)

Après capture des CSV, le flux nominal est :

1. **Entraîner** un modèle sur `data/raw/*.csv`
2. **Prédire en live** : le script capture 1,5 s via le MPU6050 et affiche `VERTICAL` / `HORIZONTAL` / `ROND`

```bash
pip install -r requirements.txt

# (re)entraîner — écrit artifacts/gesture_classifier.joblib
python python/train_model.py

# classification live (Arduino flashé + branché)
python python/predict_live.py --port COM3          # Windows
python python/predict_live.py --port /dev/ttyUSB0  # Linux
```

`predict_live.py` : compte à rebours → envoie `s` → 75 échantillons → features → prédiction (+ probabilités). Boucle jusqu’à Ctrl+C (`--once` pour une seule prise).

Un modèle pré-entraîné (RandomForest, ~97 % test sur les 600 essais) est versionné dans `artifacts/`. Relancer `train_model.py` si vous ajoutez des données.

## Notebooks ML (exploration / évaluation)

| Notebook | Rôle |
|----------|------|
| [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | Exploration : essais, fréquences, plots, stats |
| [`notebooks/02_train_eval.ipynb`](notebooks/02_train_eval.ipynb) | Comparaison modèles + métriques détaillées |

Helper : [`python/gesture_dataset.py`](python/gesture_dataset.py). Scripts CLI : [`python/train_model.py`](python/train_model.py), [`python/predict_live.py`](python/predict_live.py).

```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_train_eval.ipynb
```
