# 3D Creature Generation via Skeleton-Conditioned Optimal Transport Flow Matching

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Three.js](https://img.shields.io/badge/Three.js-r170-black.svg)](https://threejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Space-Demo-yellow)](https://huggingface.co/spaces/MissawB/creature-flow-viewer)
[![Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Collection-model%20%C2%B7%20dataset-yellow)](https://huggingface.co/collections/MissawB/skeleton-conditioned-creature-flow-6ad8fe174913aa4ba69dd554)

Générer une créature 3D (nuage de points) **à partir de son seul squelette**, puis composer des **chimères** en greffant des morceaux de squelette d'une espèce sur une autre. Le modèle est un **Flow Matching à transport optimal (OT-FM)** avec un backbone **Point-DiT**, écrit en PyTorch pur.

![Créatures générées à partir de squelettes : vues de profil et de dessus](docs/images/creatures.png)

*Nuages de 2048 points générés en 10 pas d'Euler à partir du squelette seul (en rouge). La dernière colonne est une chimère : corps de lion, cou et tête d'équidé.*

**[Essayer la démo](https://huggingface.co/spaces/MissawB/creature-flow-viewer)** · **[Méthode et résultats détaillés](docs/resultats.md)**

---

## Idées clés

- **Couplage OT par forme** : pour chaque créature, les 2048 points de bruit sont appariés aux 2048 points de surface (affectation exacte, ou Sinkhorn sur GPU). La cible $x_1 - x_0$ devient presque déterministe (loss de 1.25 à 0.013), la surface s'affine et les trajectoires sont presque droites : **10 pas d'Euler** suffisent.
- **Cross-attention points → squelette avec biais de distance multi-échelle** : chaque tête d'attention a sa propre portée ($\lambda_h = 2 \cdot 2^h$, de 2 à 256), calculée sur une estimation de la forme finale plutôt que sur le bruit. C'est ce qui fait passer la reconnaissance de famille de 40 % à 92 %.
- **Le squelette doit porter l'information** : une famille mal reconnue l'était parce que son squelette ne la distinguait pas. Corriger les données (écartement des pattes, port de tête) a fait plus que n'importe quel changement de modèle.
- **Greffe cinématique** : un sous-arbre du squelette donneur (cou et tête, par exemple) est recollé sur l'hôte, et le modèle génère la chimère d'un seul tenant.

## Résultats

Modèle de 10,4 M de paramètres, 500 quadrupèdes procéduraux (5 familles, squelette de 35 articulations inspiré de SMAL), 180 époques en bf16, **environ 32 min sur une RTX 4060 Laptop**. Évaluation sur des squelettes jamais vus, 200 formes générées contre 200 réelles :

| Mesure | Modèle final | Repère |
|---|---|---|
| Famille reconnue (40 squelettes) | **100 %** | 40 % pour la première version |
| CD vers sa famille | **0.0009** | 0.0007 entre deux vraies formes |
| Écart médian à la surface | **0.0029** | 0.0129 avant OT et mise à l'échelle |
| MMD-CD / COV-CD | **0.00084 / 57 %** | |

Ce qui a compté, dans l'ordre : le biais multi-échelle, des squelettes qui distinguent les familles, la mise à l'échelle des données vers celle du bruit (`data_scale: 4`), un entraînement plus long que ce que la loss laissait croire, puis le couplage OT. Les ablations, ce qui n'a pas marché (Fourier features, modèle élargi, autoguidage, Not-So-OT…) et la lecture du 1-NNA sont dans **[docs/resultats.md](docs/resultats.md)**.

**Limites principales** : les formes sont des cylindres autour d'un squelette paramétrique, pas de vrais animaux, et les 5 familles sont séparées par construction ; le 100 % mesure que le modèle suit le squelette, pas qu'il généralise. La surface générée reste plus diffuse que la vraie (0.0029 contre 0.0004).

## En cours : toutes les espèces

L'étape 2 remplace les quadrupèdes procéduraux par **40 000 créatures réelles d'[Articulation-XL2.0](https://huggingface.co/datasets/Seed3D/Articulation-XL2.0)** (animaux, créatures mythiques, personnages), avec des squelettes de topologie libre. Les données sont prêtes ([rapport](docs/axl/report.md)) ; le modèle à squelette libre (encodeur de graphe, code de forme, OT sur GPU) est en cours de mise au point. Feuille de route : [conception](docs/conception.md) ; état de l'art : [docs/etat-de-l-art.md](docs/etat-de-l-art.md).

---

## Démarrage rapide

```bash
python -m venv .venv
# Linux/macOS : source .venv/bin/activate   —   Windows : .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU NVIDIA ; omettre --index-url pour la version CPU
pip install -r requirements.txt
pytest tests/ -v                                   # dont un test d'overfit (~10 s sur CPU)
```

### Quadrupèdes (étape 1)
```bash
python scripts/prepare_data.py --config configs/default_config.yaml
python scripts/train.py --config configs/default_config.yaml
python scripts/evaluate_families.py                # reconnaissance de famille
python scripts/evaluate_generation.py              # MMD, COV, 1-NNA, écart à la surface (~4 min)
python scripts/sample_and_export.py                # export pour la visionneuse
python -m http.server 8000 --directory viewer      # puis http://localhost:8000
```

Dans la visionneuse, le curseur $t \in [0, 1]$ montre la créature émerger du bruit.

### Toutes espèces (étape 2)
```bash
python scripts/build_axl_cache.py --split train --workers 8
python scripts/build_axl_cache.py --split test
python scripts/train_axl.py --config configs/axl_config.yaml
python scripts/evaluate_axl.py --checkpoint <checkpoint.pt>
```

## Organisation du code

```text
configs/     hyperparamètres (default_config.yaml : quadrupèdes, axl_config.yaml : toutes espèces)
data/        générateur procédural SMAL-like, couplage OT, données Articulation-XL (data/axl)
models/      Point-DiT, encodeurs de squelette, flow matching, OT sur GPU, modèle à squelette libre
pipeline/    entraînement, EMA, greffe de chimères, métriques
scripts/     préparation des données, entraînement, évaluation, export
viewer/      visionneuse Three.js
docs/        résultats détaillés, conception, état de l'art, rapport Articulation-XL
```
