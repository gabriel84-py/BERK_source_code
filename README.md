# BERK — Robot Autonome de Localisation

> **B**ot d'**E**xploration avec **R**epérage par fusion de capteurs et vision par ordinateur

[![Licence: GPL v3](https://img.shields.io/badge/Licence-GPLv3-blue.svg)](LICENSE.md)
[![MicroPython](https://img.shields.io/badge/MicroPython-Pico%202W-orange.svg)](https://micropython.org/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-green.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-profondeur-red.svg)](https://pytorch.org/)

---

## Présentation

BERK est un robot autonome capable d'estimer sa position en temps réel dans un environnement inconnu, en combinant **quatre méthodes de localisation** :

| Méthode | Erreur moyenne | Stabilité (σ) |
|---|---|---|
| Odométrie seule | 13,86 cm | 4,79 |
| Ultrasons | 4,77 cm | 1,58 |
| Vision (IA profondeur) | 1,98 cm | 0,30 |
| **Fusion (Filtre de Kalman)** | **0,80 cm** | **0,40** |

**Problématique** : *Comment améliorer la précision de la localisation et du déplacement d'un robot autonome en combinant différentes sources de données dans un environnement inconnu ?*

---

## Architecture matérielle

```
┌────────────────────────────────────────────┐
│           RASPBERRY PI 4B (cerveau)        │
│  - OS Linux                                │
│  - Fusion des données (Filtre de Kalman)   │
│  - Interface web (FastAPI)                 │
│  - Modèle deep learning (carte profondeur) │
│  - Commandes vers le Pico                  │
└──────────────────┬─────────────────────────┘
                   │ USB série (115 200 baud)
                   │ Données : C/L/R/TL/TR
                   │ Commandes : AVANCE/RECUL/PIVOT_G/PIVOT_D/STOP
                   ▼
┌────────────────────────────────────────────┐
│       RASPBERRY PI PICO 2W (exécutant)     │
│  - MicroPython                             │
│  - 3× capteurs ultrasoniques (HC-SR04)     │
│  - 2× moteurs CC avec encodeurs (4200 t/t) │
│  - Transmission des données capteurs 10 Hz │
└────────────────────────────────────────────┘
```

**Châssis** : PETG imprimé en 3D  
**Alimentation** : LiPo 3S 12,5V — 2300 mAh  
**Caméra** : Raspberry Pi Camera (120° FOV)

---

## Méthodes de localisation

### 1. Odométrie
Estimation de position à partir des encodeurs magnétiques des moteurs.

```
d = (nbTicks / nbTicksParTour) × π × diamètreRoue
```
- Fréquence : 100 Hz
- Limite : dérive cumulée (glissement, jeu mécanique)

### 2. Ultrasons
3 capteurs (centre, gauche, droite) pour la détection d'obstacles et la correction de cap.

### 3. Vision par ordinateur (IA)
Modèle de deep learning maison (PyTorch) produisant une **carte de profondeur** à partir du flux vidéo. Le modèle :
- Détecte les bords du mobilier
- Identifie les objets (chaise, table…)
- Calcule la distance par analyse des perspectives (angle de vue 120°)

### 4. Fusion — Filtre de Kalman
Inventé en 1960 par Rudolf E. Kalman, utilisé pour la première fois dans le programme Apollo.

```
État initial :  X=0, P=1.0, Q=42, R=5

Étape 1 — Prédiction :
  X_prédit = X_précédent
  P_prédit = P_précédent + Q          (Q=42 : bruit odomètre)

Étape 2 — Gain de Kalman :
  K = P_prédit / (P_prédit + R)       (R=5 : bruit capteur)
  K → 0 : confiance prédiction
  K → 1 : confiance mesure

Étape 3 — Mise à jour :
  X_nouveau = X_prédit + K × (Mesure − X_prédit)

Étape 4 — Mise à jour incertitude :
  P_nouveau = (1 − K) × P_prédit
```

La **mesure** est la moyenne entre ultrasons (±4 cm) et vision (±6 cm).  
La **prédiction** est fournie par l'odométrie (erreur ≈ 13 cm²).

---

## Structure du projet

```
berk/
├── sources/
│   ├── main.py          # Serveur FastAPI + fusion + threads (Raspberry Pi 4B)
│   ├── pico.py          # Firmware MicroPython (Raspberry Pi Pico 2W)
│   ├── graphes_gen.py   # Génération des graphiques d'analyse
│   ├── robot_data.json  # Données expérimentales collectées
│   └── aruco.py         # Génération de marqueurs ArUco (exploratoire)
├── Model/
│   ├── run.py           # Entrée du modèle de profondeur
│   └── model/           # Architecture du réseau de neurones (PyTorch)
├── LICENSE.md
└── README.md
```

---

## Installation & Démarrage

### Prérequis

**Raspberry Pi 4B :**
```bash
pip install fastapi uvicorn picamera2 opencv-python numpy pillow pyserial torch
```

**Raspberry Pi Pico 2W :**
- Flasher MicroPython 2.x
- Copier `sources/pico.py` sur la Pico

### Lancer le serveur

```bash
cd sources
python main.py
```

L'interface web est accessible sur `http://<ip-du-robot>:8000`

---

## Interface Web

L'interface (FastAPI + HTML/JS) permet de :

- **Visualiser** le flux caméra en temps réel avec overlay
- **Piloter** le robot (touches fléchées ou boutons)
- **Comparer** les 4 méthodes de localisation en temps réel
- **Collecter des données** expérimentales (3 modes)
- **Exporter** `robot_data.json` pour analyse Python

### API REST

| Endpoint | Méthode | Description |
|---|---|---|
| `/state` | GET | État complet (capteurs + positions) |
| `/cmd` | POST | Commande moteur |
| `/reset_all` | POST | Réinitialiser toutes les méthodes |
| `/exp/start` | POST | Démarrer une session d'expérience |
| `/exp/snapshot` | GET | Capture instantanée des 4 méthodes |
| `/exp/record` | POST | Enregistrer une mesure avec position réelle |
| `/exp/export` | GET | Télécharger `robot_data.json` |
| `/video` | GET | Flux MJPEG caméra |

---

## Analyse des données

Le script `graphes_gen.py` génère 4 graphiques à partir de `robot_data.json` :

```bash
cd sources
python graphes_gen.py
```

Les exports sont sauvegardés dans `sources/exports/`.

**Graphiques produits :**
- `erreur_moyenne_*.png` — Précision par méthode
- `erreur_temps_*.png` — Dérive temporelle
- `stabilite_*.png` — Écart-type (stabilité)
- `trajectoire_*.png` — Trajectoire réelle vs estimée

---

## 🔬 Méthodologie expérimentale

Pour chaque méthode, **3 types d'expériences × 5 essais** :

1. **Mesures de position** — erreur entre position réelle (mètre ruban) et estimée
2. **Mesures dans le temps** — observation de la dérive sur 4 secondes
3. **Trajectoire** — comparaison du trajet réel et estimé tous les 20 cm

Le repère est matérialisé au sol avec des bandes adhésives sur surface plane.

---

## Limites connues

- Mesures réelles effectuées manuellement (incertitude humaine ~0,5 cm)
- Seulement 5 essais par méthode (conditions peu contrôlées)
- Capteurs grand public (vs capteurs industriels)
- Performances visuelles dépendantes des conditions d'éclairage

---

## Remerciements

- [Fondation Raspberry Pi](https://www.raspberrypi.org/) — accessibilité de l'électronique
- [MicroPython](https://micropython.org/) — firmware Pico
- [Marcel Winheim](https://www.thingiverse.com/) — enclosure caméra open source
- [Picamera2](https://github.com/raspberrypi/picamera2) — lecture flux vidéo
- [PyTorch](https://pytorch.org/) — modèle de carte de profondeur

---

## Auteurs

**Projet BERK** — Gabriel Jean Vermeille, Amine Akachar, Timoté Jaga Bouhamouche

> L'intégralité du code a été écrit par les auteurs.

---

## Licence

Ce projet est distribué sous licence **GPL v3+** — voir [LICENSE.md](LICENSE.md).
