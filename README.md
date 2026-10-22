# 3D Creature Generation via Skeleton-Conditioned Optimal Transport Flow Matching

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Three.js](https://img.shields.io/badge/Three.js-r170-black.svg)](https://threejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Space-Demo-yellow)](https://huggingface.co/spaces/MissawB/creature-flow-viewer)
[![Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Collection-model%20%C2%B7%20dataset-yellow)](https://huggingface.co/collections/MissawB/skeleton-conditioned-creature-flow-6ad8fe174913aa4ba69dd554)

Projet de recherche et portfolio en Deep Learning géométrique 3D : génération de nuages de points de quadrupèdes conditionnée par un squelette cinématique ($K=35$ articulations), et synthèse d'hybrides chimériques par greffe de sous-arbres du squelette. Le modèle génératif est un **Flow Matching à chemins de transport optimal conditionnel (OT-FM)** avec un backbone **Point-DiT**.

![Créatures générées à partir de squelettes : vues de profil et de dessus](docs/images/creatures.png)

*Nuages de 2048 points générés en 20 pas de Heun à partir du squelette seul (en rouge). La dernière colonne est une chimère : corps de lion, cou et tête d'équidé.*

---

## 🌟 Points Forts du Projet

- **Optimal Transport Flow Matching** : chemins linéaires $x_t = (1-t)x_0 + t x_1$ entre le bruit gaussien et la surface de la créature, échantillonnage en 15 à 20 pas d'ODE.
- **Point-DiT** : modulation temporelle AdaLN-Zero, self-attention sur les points, et **cross-attention spatiale points $\to$ squelette** avec un biais de distance multi-échelle (voir [Conditionnement par le squelette](#-conditionnement-par-le-squelette)).
- **Squelette cinématique à 35 articulations inspiré de SMAL**, pour 5 familles : félins, canidés, équidés, bovidés, hippopotamidés.
- **Greffe cinématique d'hybrides** : un sous-arbre du squelette donneur (par ex. cou et tête) est recollé sur l'hôte au niveau de l'articulation de jonction.
- **Moyenne mobile des poids (EMA)** pour la génération, et **guidage sans classifieur (CFG)** optionnel.
- **Visionneuse Web 3D (Three.js)** : inspection des nuages de points et du squelette, et animation du flux $t = 0 \to 1$.
- **100 % PyTorch** : pas d'extension C++/CUDA à compiler (ni PyTorch3D ni DGL).

---

## 📐 Formulation Mathématique

### 1. Interpolation par Transport Optimal
Pour $x_0 \sim \mathcal{N}(0, I)$ et $x_1 \sim p_{\text{data}}(x \mid S)$ :
$$x_t = (1 - t) x_0 + t x_1, \quad t \in [0, 1]$$
$$u_t(x_t \mid x_0, x_1) = \frac{dx_t}{dt} = x_1 - x_0$$

Le couplage entre $x_0$ et $x_1$ est indépendant (chemin OT *conditionnel*, au sens de Lipman et al., 2023) ; il n'y a pas de couplage OT par mini-batch.

### 2. Objectif d'Apprentissage
$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, x_0, (x_1, S)} \left[ \| v_\theta(x_t, t, S) - (x_1 - x_0) \|^2 \right]$$

Le réseau traite les points comme un ensemble (équivariance par permutation) : il ne peut apprendre que le champ de vitesse marginal, dont la variance est irréductible. La loss plafonne donc autour de 0.3 même quand le modèle reproduit bien les formes ; la Chamfer Distance est le critère pertinent.

### 3. Solveurs ODE d'Inférence
- **Euler** : $x_{t + \Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t, S)$
- **Heun (ordre 2)** : prédiction-correction, utilisé par défaut.

---

## 🦴 Conditionnement par le Squelette

Chaque point interroge les 35 articulations encodées par un Transformer sur le squelette. Le score d'attention de la tête $h$ reçoit un biais de distance :

$$\text{score}_{h}(i, k) = \frac{q_i \cdot k_k}{\sqrt{d}} - \lambda_h \, \| \hat{x}_{1,i} - J_k \|^2, \qquad \lambda_h = \lambda_0 \cdot 2^h$$

- **Multi-échelle** : avec $\lambda_0 = 2$ (`spatial_distance_scale` dans la config), les 8 têtes couvrent des échelles de 2 à 256, de l'attention presque globale à l'attention très locale. Les $\lambda_h$ sont paramétrés en log et apprenables, mais en pratique ils restent proches de leur initialisation.
- **Estimation de la forme finale** : la distance est calculée sur une estimation $\hat{x}_1 = x_t + (1-t)\,\hat{v}$ affinée après chaque bloc par une tête linéaire initialisée à zéro, plutôt que sur $x_t$, qui n'est que du bruit au début de la trajectoire.

---

## 📊 Résultats

Modèle de 10,4 M de paramètres entraîné 30 époques sur 500 créatures (100 par famille), soit environ 16 min sur une RTX 4060 Laptop (pic mémoire de 5,6 Go). La génération utilise une moyenne mobile exponentielle des poids (EMA, décroissance 0.995, soit un horizon d'environ 6 époques) plutôt que leur dernier état.

**Protocole** (`scripts/evaluate_families.py`) : 40 squelettes jamais vus (8 par famille, seed disjointe de l'entraînement). Chaque créature générée est attribuée à la famille dont les 8 formes réelles de référence sont en moyenne les plus proches en Chamfer Distance.

| Variante | Biais de distance | Position utilisée | Famille reconnue | CD vers sa famille |
|---|---|---|---|---|
| Biais fixe (version initiale) | $\lambda = 1.5$ | $x_t$ | 16/40 (40 %) | 0.0060 |
| Estimation $\hat{x}_1$ seule | $\lambda = 1.5$ | $\hat{x}_1$ | 10/40 (25 %) | 0.0076 |
| Multi-échelle seul | $2 \to 256$ | $x_t$ | 31/40 (78 %) | 0.0031 |
| Multi-échelle + $\hat{x}_1$, sans EMA (3 seeds) | $2 \to 256$ | $\hat{x}_1$ | 89 % (82 à 92 %) | 0.0028 |
| **Modèle retenu** : idem + EMA (3 seeds) | $2 \to 256$ | $\hat{x}_1$ | **92 %** (90 à 95 %) | **0.0027** |

À titre de repère, la Chamfer Distance entre deux créatures réelles d'une même famille va de 0.0005 à 0.0012 (0.0007 en moyenne), et celle d'un modèle non entraîné d'environ 1.9.

**Lecture de l'ablation** : le biais multi-échelle explique l'essentiel du gain. Avec un $\lambda$ unique et faible, le biais est presque plat à l'échelle des créatures et le modèle produit un « quadrupède moyen ». L'estimation $\hat{x}_1$ n'apporte rien seule. Combinée au multi-échelle, elle semble ajouter environ 10 points (78 % sur un run, contre 82 à 92 % sur trois runs du modèle final) ; les variantes intermédiaires n'ont qu'un run chacune, donc ce gain reste à confirmer.

**EMA** : sur 3 runs, les poids moyennés donnent 95 / 90 / 92 %, contre 85 / 82 / 92 % pour le dernier état des poids des mêmes runs. Le gain est modeste (+6 points en moyenne) et réduit surtout les mauvais runs ; la précision des formes ne change pas.

### Ce que le squelette doit montrer

Le modèle ne voit que le squelette. Les bovidés et les équidés se distinguaient surtout par l'épaisseur du corps, qu'aucune articulation n'encodait : le modèle générait des bovidés presque aussi fins que des équidés (rayon médian du torse 0.187, contre 0.238 pour les vraies formes), et la moitié étaient reconnus comme équidés. Les deux dernières versions du générateur corrigent les données, pas le modèle :

| Générateur | Changement | Famille reconnue (3 seeds) | Bovidés | CD vers sa famille |
|---|---|---|---|---|
| v2 | (modèle retenu ci-dessus) | 92 % (90 à 95 %) | 17/24 | 0.0027 |
| v3 | pattes posées sur un même sol | 90 % (90 / 90 / 90) | 13/24 | 0.0026 à 0.0027 |
| **v4** | écartement des pattes proportionnel à l'épaisseur du corps ; cou des bovidés incliné à 20° (49° pour les autres) | **100 %** (40/40 sur chaque seed) | **24/24** | **0.0026 à 0.0028** |

Sur les vraies formes, l'écart de Chamfer entre un bovidé et la famille la plus proche passe de 0.0020 à 0.0044 : il dépasse maintenant l'erreur du modèle (0.0027), alors qu'il était en dessous. Les autres familles ne perdent rien. Doubler les données (200 formes par famille, générateur v3, un run) réduit la CD à 0.0019 mais laisse 3 bovidés sur 8 confondus : la quantité de données affine les formes, elle ne remplace pas une information absente du squelette.

Le modèle publié et la visionneuse utilisent le générateur v4 (seed 0).

---

## ⚠️ Limites

- **Données procédurales, pas SMAL** : les formes sont des cylindres échantillonnés autour d'un squelette paramétrique inspiré de la topologie SMAL, pas le modèle SMAL lui-même (`shape_betas` est inutilisé). La variabilité intra-famille est faible (±3 % sur les proportions).
- **100 % de reconnaissance, sur un test facile** : avec le générateur v4, les 5 familles sont séparées par construction (proportions, écartement, port de tête) et leur variabilité interne est faible ; ce score mesure que le modèle suit le squelette, pas qu'il généralise à de vrais animaux.
- **Précision** : les formes générées restent environ 4 fois plus éloignées de leur famille que deux créatures réelles entre elles (0.0027 contre 0.0007).
- **Greffe** : la jonction aligne le sous-arbre donneur par translation, sans mise à l'échelle ni rotation.
- **Une seule seed par variante intermédiaire** dans l'ablation. Le modèle final varie de 90 à 95 % selon la seed (3 runs), et l'évaluation sur 40 squelettes a elle-même une incertitude d'environ ±8 points.
- **Le couplage OT par forme n'aide pas encore** (`ot_noise_per_sample` dans la config, désactivé par défaut) : il rend l'échantillonnage en 2 à 5 pas bien meilleur (CD 0.0063 en 2 pas, contre 0.049), mais fait tomber la reconnaissance de famille à 42 % en 20 pas, avec 20 % comme avec 50 % de bruit indépendant. Le guidage sans classifieur (`cond_drop_prob: 0.1` à l'entraînement, `guidance_scale` à la génération) remonte ce score à 55–60 % pour $w$ entre 1.5 et 2 (un seul run), encore loin du modèle final ; à partir de $w = 2.5$, les formes se dégradent.

---

## 📁 Architecture du Répertoire

```text
3D_Creation/
├── configs/
│   └── default_config.yaml       # Hyperparamètres (modèle, entraînement, dataset)
├── data/
│   ├── smal_quadrupeds.py        # Hiérarchie 35 joints & générateur procédural de surfaces
│   └── dataset.py                # Dataset PyTorch, cache versionné & augmentations
├── models/
│   ├── point_dit.py              # Point-DiT, AdaLN-Zero, cross-attention spatiale multi-échelle
│   ├── skeleton_encoder.py       # Encodeur Transformer du squelette
│   └── flow_matching.py          # OT-Flow Matching & solveurs ODE (Euler / Heun)
├── pipeline/
│   ├── hybrid_splicer.py         # Greffe de sous-arbres pour les chimères
│   ├── metrics.py                # Chamfer Distance, couverture du squelette, nouveauté
│   └── trainer.py                # Boucle d'entraînement CPU/GPU
├── viewer/
│   ├── index.html                # Visionneuse 3D HTML5
│   ├── app.js                    # Contrôles Three.js (caméra, curseur t, rendu)
│   ├── style.css                 # Thème « bocal de formol »
│   ├── favicon.svg               # Icône (squelette ocre)
│   └── data/                     # Trajectoires exportées (creatures.json)
├── scripts/
│   ├── prepare_data.py           # Génération et mise en cache du dataset
│   ├── train.py                  # Entraînement
│   ├── sample_and_export.py      # Génération de trajectoires & export JSON pour la visionneuse
│   ├── plot_creatures.py         # Image du README à partir des créatures exportées
│   ├── evaluate.py               # Chamfer Distance & couverture du squelette
│   └── evaluate_families.py      # Reconnaissance de famille & matrice de confusion
├── tests/                        # Tests unitaires et test d'overfit (pytest)
├── docs/                         # Spécifications, plans et images
├── requirements.txt
└── README.md
```

---

## 🚀 Guide de Démarrage Rapide

### 1. Installation de l'environnement
```bash
python -m venv .venv
# Linux/macOS : source .venv/bin/activate   —   Windows : .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU NVIDIA ; omettre --index-url pour la version CPU
pip install -r requirements.txt
```

### 2. Exécution des tests
```bash
pytest tests/ -v
```
`tests/test_overfit.py` vérifie que le modèle apprend réellement une créature (environ 10 s sur CPU).

### 3. Génération du dataset
```bash
python scripts/prepare_data.py --config configs/default_config.yaml
```

### 4. Entraînement
```bash
python scripts/train.py --config configs/default_config.yaml
```

### 5. Évaluation
```bash
python scripts/evaluate.py
python scripts/evaluate_families.py
```

### 6. Export des créatures pour la visionneuse
```bash
python scripts/sample_and_export.py                                        # dernier checkpoint de la config
python scripts/sample_and_export.py --checkpoint checkpoints/<run>/model_epoch_030.pt
python scripts/plot_creatures.py                                          # régénère docs/images/creatures.png
```

### 7. Lancement de la visionneuse 3D
```bash
python -m http.server 8000 --directory viewer
```
Puis ouvrez `http://localhost:8000`. La caméra se manipule à la souris, et le curseur temporel $t \in [0, 1]$ montre la créature émerger du bruit gaussien.
