# 3D Creature Generation via Skeleton-Conditioned Optimal Transport Flow Matching

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Three.js](https://img.shields.io/badge/Three.js-r128-black.svg)](https://threejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Projet de recherche et portfolio en Deep Learning géométrique 3D pour la génération de créatures 3D anatomiquement plausibles et la synthèse d'hybrides chimériques, guidés par un squelette cinématique standardisé ($K=35$ articulations) et un modèle génératif **Optimal Transport Flow Matching (OT-FM)**.

---

## 🌟 Points Forts du Projet

- **Optimal Transport Flow Matching (OT-FM)** : Trajectoires de vitesse quasi-linéaires reliant le bruit gaussien $\mathcal{N}(0, I)$ à la surface de la créature, permettant un échantillonnage ultra-rapide (15 à 20 pas ODE).
- **Point-DiT (Diffusion/Flow Transformer)** : Backbone moderne intégrant la modulation temporelle AdaLN (*Adaptive Layer Norm*) et un bloc de **Cross-Attention spatiale (Point $\to$ Squelette)** avec biais géométrique de distance euclidienne.
- **Squelette cinématique SMAL (35 articulations)** : Représentation hiérarchique standardisée des quadrupèdes (félins, canidés, équidés, bovidés, hippopotamidés).
- **Greffe cinématique d'hybrides (Subtree Splicing)** : Composition de créatures chimériques (ex. corps de lion + tête/cou d'équidé) avec continuité anatomique garantie aux jonctions.
- **Visionneuse Web 3D interactive (Three.js)** : Application WebGL autonome permettant d'inspecter les nuages de points 3D, le squelette articulé et d'animer le flux temporel $t=0 \to 1$ en direct.
- **Zéro dépendance C++/CUDA instable** : Implémentation 100% native en PyTorch vectorisé, sans dépendance externe lourde (pas besoin de compiler PyTorch3D ou DGL).

---

## 📐 Formulation Mathématique

### 1. Interpolation par Transport Optimal
Pour $x_0 \sim \mathcal{N}(0, I)$ et $x_1 \sim p_{\text{data}}(x \mid S)$ :
$$x_t = (1 - t) x_0 + t x_1, \quad t \in [0, 1]$$
$$u_t(x_t \mid x_0, x_1) = \frac{dx_t}{dt} = x_1 - x_0$$

### 2. Objectif d'Apprentissage
$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, x_0, (x_1, S)} \left[ \| v_\theta(x_t, t, S) - (x_1 - x_0) \|^2 \right]$$

### 3. Solveurs ODE d'Inférence
- **Euler** : $x_{t + \Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t, S)$
- **Heun (ordre 2)** : Prédiction-correction pour des trajectoires encore plus nettes.

---

## 📁 Architecture du Répertoire

```text
3D_Creation/
├── configs/
│   └── default_config.yaml       # Hyperparamètres (modèle, entraînement, dataset)
├── data/
│   ├── smal_quadrupeds.py        # Hiérarchie 35 joints SMAL & générateur paramétrique
│   └── dataset.py                # Dataset PyTorch & augmentations géométriques
├── models/
│   ├── point_dit.py              # Architecture Point-DiT & blocs modulés AdaLN
│   ├── skeleton_encoder.py       # Encodeur cinématique (Self-Attention sur squelette)
│   └── flow_matching.py          # OT-Flow Matching & solveurs ODE (Euler / Heun)
├── pipeline/
│   ├── hybrid_splicer.py         # Greffe et interpolation de squelettes hybrides
│   ├── metrics.py                # Chamfer Distance vectorisée, Skel-coverage, Nouveauté
│   └── trainer.py                # Boucle d'entraînement PyTorch avec gestion CPU/GPU
├── viewer/
│   ├── index.html                # Visionneuse 3D HTML5
│   ├── app.js                    # Contrôles Three.js (caméra, slider t, rendu 3D)
│   ├── style.css                 # Interface moderne sombre glassmorphism
│   └── data/                     # Trajectoires exportées (creatures.json)
├── scripts/
│   ├── prepare_data.py           # Génération et mise en cache du dataset SMAL
│   ├── train.py                  # Script d'entraînement principal
│   ├── sample_and_export.py      # Génération de trajectoires & export JSON pour viewer
│   └── evaluate.py               # Évaluation des métriques quantitatives (CD, Coverage)
├── tests/                        # Suite complète de tests unitaires (pytest)
├── docs/superpowers/             # Spécifications et plans de conception détaillés
├── requirements.txt              # Dépendances légères (torch, numpy, scipy, pyyaml, tqdm)
└── README.md
```

---

## 🚀 Guide de Démarrage Rapide

### 1. Installation de l'environnement
```bash
pip install -r requirements.txt
```

### 2. Exécution des tests unitaires
```bash
pytest tests/ -v
```

### 3. Génération du dataset synthétique SMAL
```bash
python scripts/prepare_data.py --config configs/default_config.yaml
```

### 4. Entraînement du modèle Point-DiT
```bash
python scripts/train.py --config configs/default_config.yaml
```

### 5. Génération et export des créatures pour la visionneuse
```bash
python scripts/sample_and_export.py
```

### 6. Lancement de la Visionneuse 3D
Ouvrez simplement un serveur HTTP local léger dans le dossier `viewer/` :
```bash
python -m http.server 8000 --directory viewer
```
Puis accédez à `http://localhost:8000` dans votre navigateur.
Vous pouvez manipuler la caméra à la souris, faire défiler le flux génératif via le curseur temporel $t \in [0, 1]$, et observer la créature émerger du bruit gaussien jusqu'à sa forme anatomique complète !
