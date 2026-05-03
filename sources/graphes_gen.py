import json
import os
import math
import matplotlib.pyplot as plt
from datetime import datetime

"""
==============================
ROBOT ANALYSIS TOOLKIT
==============================

Dossier de sortie : ./exports/
Fichier de données : robot_data.json
"""

DATA_FILE = "robot_data.json"
EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

# ----------------------
# Data management
# ----------------------

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return {"methods": {}, "time_series": {}, "trajectories": {}}


def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# ----------------------
# Math
# ----------------------

def position_error(xr, yr, xm, ym):
    return math.sqrt((xm - xr)**2 + (ym - yr)**2)


def mean(vals):
    return sum(vals) / len(vals) if vals else 0


def std_dev(vals):
    m = mean(vals)
    return math.sqrt(sum((v - m)**2 for v in vals) / len(vals)) if vals else 0

# ----------------------
# Input
# ----------------------

def add_measurement(data):
    print("Ajout mesure position (erreur calculée automatiquement)")
    m = input("Méthode (odom/ultrason/vision/fusion): ")
    xr = float(input("x réel (cm): "))
    yr = float(input("y réel (cm): "))
    xm = float(input("x estimé (cm): "))
    ym = float(input("y estimé (cm): "))
    e = position_error(xr, yr, xm, ym)
    data["methods"].setdefault(m, []).append(e)
    print(f"Erreur = {e:.2f} cm")


def add_time_series(data):
    print("Ajout point (temps, erreur)")
    m = input("Méthode: ")
    t = float(input("Temps (s): "))
    e = float(input("Erreur (cm): "))
    data["time_series"].setdefault(m, []).append((t, e))


def add_trajectory(data):
    print("Ajout point de trajectoire (réel + estimé)")
    m = input("Méthode: ")
    xr = float(input("x réel: "))
    yr = float(input("y réel: "))
    xm = float(input("x estimé: "))
    ym = float(input("y estimé: "))
    data["trajectories"].setdefault(m, []).append((xr, yr, xm, ym))

# ----------------------
# Plots + export
# ----------------------

def save_plot(fig, name):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(EXPORT_DIR, f"{name}_{ts}.png")
    fig.savefig(path)
    print(f"Graph sauvegardé: {path}")


def plot_bar(data):
    methods, means = [], []
    for m, vals in data["methods"].items():
        methods.append(m)
        means.append(mean(vals))
    fig = plt.figure()
    plt.bar(methods, means)
    plt.title("Erreur moyenne par méthode")
    plt.xlabel("Méthode")
    plt.ylabel("Erreur (cm)")
    save_plot(fig, "erreur_moyenne")
    plt.show()


def plot_time(data):
    fig = plt.figure()
    for m, vals in data["time_series"].items():
        vals = sorted(vals)
        t = [v[0] for v in vals]
        e = [v[1] for v in vals]
        plt.plot(t, e, label=m)
    plt.title("Erreur vs Temps")
    plt.xlabel("Temps (s)")
    plt.ylabel("Erreur (cm)")
    plt.legend()
    save_plot(fig, "erreur_temps")
    plt.show()


def plot_std(data):
    methods, stds = [], []
    for m, vals in data["methods"].items():
        methods.append(m)
        stds.append(std_dev(vals))
    fig = plt.figure()
    plt.bar(methods, stds)
    plt.title("Stabilité (écart-type)")
    plt.xlabel("Méthode")
    plt.ylabel("Écart-type")
    save_plot(fig, "stabilite")
    plt.show()


def plot_trajectory(data):
    fig = plt.figure()
    for m, pts in data["trajectories"].items():
        xr = [p[0] for p in pts]
        yr = [p[1] for p in pts]
        xm = [p[2] for p in pts]
        ym = [p[3] for p in pts]
        plt.plot(xr, yr, linestyle='--', label=f"{m} réel")
        plt.plot(xm, ym, label=f"{m} estimé")
    plt.title("Trajectoire réelle vs estimée")
    plt.xlabel("X (cm)")
    plt.ylabel("Y (cm)")
    plt.legend()
    save_plot(fig, "trajectoire")
    plt.show()

# ----------------------
# Analysis text generator
# ----------------------

def generate_analysis(data):
    print("\n===== ANALYSE AUTOMATIQUE =====")
    for m, vals in data["methods"].items():
        m_mean = mean(vals)
        m_std = std_dev(vals)
        print(f"Méthode {m} : erreur moyenne = {m_mean:.2f} cm, stabilité = {m_std:.2f}")
    print("\nInterprétation :")
    print("- La méthode avec la plus petite erreur est la plus précise")
    print("- La méthode avec le plus petit écart-type est la plus stable")
    print("- Une forte augmentation dans le graphe temps indique une dérive")

# ----------------------
# Menu
# ----------------------

def menu():
    data = load_data()
    while True:
        print("""
MENU
1. Ajouter mesure
2. Ajouter donnée temps
3. Ajouter trajectoire
4. Graphe erreur moyenne
5. Graphe temps
6. Graphe stabilité
7. Graphe trajectoire
8. Générer analyse
9. Quitter
""")
        c = input("Choix: ")
        if c == "1": add_measurement(data)
        elif c == "2": add_time_series(data)
        elif c == "3": add_trajectory(data)
        elif c == "4": plot_bar(data)
        elif c == "5": plot_time(data)
        elif c == "6": plot_std(data)
        elif c == "7": plot_trajectory(data)
        elif c == "8": generate_analysis(data)
        elif c == "9": save_data(data); break
        save_data(data)

if __name__ == "__main__":
    menu()
"""

odom : 
110 45 100 50
115 48 100 50
108 52 100 50
120 40 100 50
112 47 100 50

ultrason : 
105 48 100 50
102 55 100 50
98 52 100 50
107 49 100 50
103 51 100 50

vision : 
101 49 100 50
99 52 100 50
102 50 100 50
98 51 100 50
100 48 100 50

fusion : 
100 49 100 50
101 50 100 50
99 50 100 50
100 51 100 50
100 50 100 50

menu 2 : 

odom
0 0
1 2
2 5
3 9
4 15

ultrason
0 0
1 1
2 3
3 5
4 7

vision
0 0
1 1
2 2
3 3
4 4

fusion
0 0
1 0.5
2 1
3 1.5
4 2

trajectoire : 

odom
0 0 0 0
20 0 22 -2
40 0 45 -5
60 0 70 -10
80 0 95 -20

ultrason
0 0 0 0
20 0 21 -1
40 0 42 -2
60 0 62 -3
80 0 83 -5

vision
0 0 0 0
20 0 20 1
40 0 39 1
60 0 61 0
80 0 80 -1

fusion
0 0 0 0
20 0 20 0
40 0 40 1
60 0 60 0
80 0 80 0
"""