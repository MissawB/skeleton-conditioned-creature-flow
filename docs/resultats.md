# Méthode et résultats détaillés

Ce document rassemble la formulation, les ablations et les mesures qui ne tiennent pas dans le [README](../README.md). Tous les chiffres concernent les quadrupèdes procéduraux (étape 1), sauf la section Articulation-XL.

**Sommaire** : [Formulation](#-formulation-mathématique) · [Conditionnement](#-conditionnement-par-le-squelette) · [Résultats et ablations](#-résultats) · [Articulation-XL](#-données-articulation-xl-en-cours) · [Limites](#️-limites)

---

## 📐 Formulation Mathématique

### 1. Interpolation par Transport Optimal
Pour $x_0 \sim \mathcal{N}(0, I)$ et $x_1 \sim p_{\text{data}}(x \mid S)$ :
$$x_t = (1 - t) x_0 + t x_1, \quad t \in [0, 1]$$
$$u_t(x_t \mid x_0, x_1) = \frac{dx_t}{dt} = x_1 - x_0$$

Le couplage entre $x_0$ et $x_1$ est un **transport optimal par forme** : pour chaque créature, les 2048 points de bruit sont appariés aux 2048 points de surface par une affectation exacte (algorithme hongrois, coût $\|x_0 - x_1\|^2$), dans l'esprit du couplage OT par mini-batch de Tong et al. (2023), mais à l'intérieur d'un nuage. Huit tirages de bruit apparié par forme sont précalculés une fois (`ot_noise_per_sample: 8`, environ 30 min de CPU, puis en cache).

### 2. Objectif d'Apprentissage
$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, x_0, (x_1, S)} \left[ \| v_\theta(x_t, t, S) - (x_1 - x_0) \|^2 \right]$$

Avec un couplage indépendant, chaque point de bruit est associé à un point de surface tiré au hasard : la cible $x_1 - x_0$ a une variance irréductible énorme (loss d'environ 1.3 même quand le modèle reproduit bien les formes), et le détail de surface n'en représente qu'une infime partie. L'appariement OT rend la cible presque déterministe (loss d'environ 0.013), ce qui affine la surface (voir [Couplage OT](#couplage-ot-par-forme)).

### 3. Solveurs ODE d'Inférence
- **Euler** : $x_{t + \Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t, S)$
- **Heun (ordre 2)** : prédiction-correction, deux évaluations du réseau par pas.

Avec le couplage OT, Euler en 10 pas (défaut) vaut Heun en 20 pas pour quatre fois moins d'évaluations du réseau.

---

## 🦴 Conditionnement par le Squelette

Chaque point interroge les 35 articulations encodées par un Transformer sur le squelette. Le score d'attention de la tête $h$ reçoit un biais de distance :

$$\text{score}_{h}(i, k) = \frac{q_i \cdot k_k}{\sqrt{d}} - \lambda_h \, \| \hat{x}_{1,i} - J_k \|^2, \qquad \lambda_h = \lambda_0 \cdot 2^h$$

- **Multi-échelle** : avec $\lambda_0 = 2$ (`spatial_distance_scale` dans la config), les 8 têtes couvrent des échelles de 2 à 256, de l'attention presque globale à l'attention très locale. Les $\lambda_h$ sont paramétrés en log et apprenables, mais en pratique ils restent proches de leur initialisation.
- **Estimation de la forme finale** : la distance est calculée sur une estimation $\hat{x}_1 = x_t + (1-t)\,\hat{v}$ affinée après chaque bloc par une tête linéaire initialisée à zéro, plutôt que sur $x_t$, qui n'est que du bruit au début de la trajectoire.

---

## 📊 Résultats

Modèle de 10,4 M de paramètres entraîné 180 époques sur 500 créatures (100 par famille), soit environ 32 min sur une RTX 4060 Laptop en précision mixte bf16 (`mixed_precision`, 3,6 Go ; environ 1 h 35 et 5,3 Go en fp32, pour la même qualité). Les ablations ci-dessous, jusqu'à la section sur la durée d'entraînement, ont été faites à 30 époques en fp32. La génération utilise une moyenne mobile exponentielle des poids (EMA, décroissance 0.995, soit un horizon d'environ 6 époques) plutôt que leur dernier état.

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

### Échelle des coordonnées

Les créatures sont normalisées dans la boule unité, donc leurs points ont un écart-type d'environ 0.25, alors que le bruit de départ a un écart-type de 1. Le modèle passe ainsi l'essentiel de la trajectoire à contracter le nuage, et les surfaces générées sont épaisses : les points s'écartent de la surface réelle d'environ 0.013 en médiane (0.0004 sur les vraies formes), avec un biais vers l'extérieur. Avec `data_scale: 4` (config), les points et les articulations sont multipliés par 4 à l'intérieur du flow puis divisés à la sortie (le biais de distance est corrigé de $1/4^2$), ce qui rapproche l'échelle des données de celle du bruit, à la manière du $\sigma_{data}$ d'EDM.

En plus de la reconnaissance de famille, la qualité est mesurée par les métriques standard de génération de nuages de points (`scripts/evaluate_generation.py`, PointFlow) : MMD-CD (fidélité, plus bas est mieux), COV-CD (couverture des formes réelles, plus haut est mieux) et 1-NNA-CD (50 % si générées et réelles sont indiscernables, 100 % si elles se séparent parfaitement). S'y ajoute l'écart à la surface : la distance signée de chaque point à la surface du cylindre d'os le plus proche, calculée à partir du squelette.

| Moyenne sur 3 seeds (générateur v4) | `data_scale: 1` | **`data_scale: 4`** |
|---|---|---|
| Écart médian à la surface | 0.0129 | **0.0092** |
| Points à plus de 30 % du rayon hors de la surface | 15.9 % | **8.8 %** |
| Famille reconnue | 100 % | **100 %** |
| CD vers sa famille | 0.0026 à 0.0028 | **0.0016 à 0.0017** |
| MMD-CD | 0.00224 | **0.00154** |
| COV-CD | 48 % | **51 %** |
| 1-NNA-CD | 98.4 % | **96.3 %** |

Les trois seeds à l'échelle 4 font mieux que la meilleure seed à l'échelle 1 sur chaque mesure. Pousser l'échelle à 8 (un run) dégrade le résultat par rapport à 4 (MMD 0.00176, 1-NNA 98.8 %).

### Durée d'entraînement

Même à l'échelle 4, la surface restait floue. Le diagnostic écarte l'intégration (20 ou 100 pas de Heun donnent le même écart, 0.0096 contre 0.0095) et l'encodage des coordonnées : des Fourier features sur l'entrée (`fourier_octaves`, Tancik et al. 2020, 6 ou 10 octaves) dégradent légèrement le résultat (écart 0.0107, MMD 0.0018). En revanche, les formes s'améliorent à chaque palier d'entraînement (écart 0.0158, 0.0121 puis 0.0096 aux époques 10, 20 et 30) alors que la loss ne bouge presque plus (1.30 à 1.29) : le modèle à 30 époques était sous-entraîné, et la loss ne le montrait pas.

| Moyenne sur 3 seeds (générateur v4, `data_scale: 4`) | 30 époques | **90 époques** |
|---|---|---|
| Écart médian à la surface | 0.0090 | **0.0051** |
| Points à plus de 30 % du rayon hors de la surface | 8.8 % | **2.2 %** |
| Famille reconnue | 100 % | **100 %** |
| CD vers sa famille | 0.0016 à 0.0017 | **0.0010 à 0.0011** |
| MMD-CD | 0.00150 | **0.00098** |
| COV-CD | 50 % | **56 %** |
| 1-NNA-CD | 99.6 % | 53.6 % (51.2 à 57.5 %) |
| &nbsp;&nbsp;dont générées bien classées | 99 % | 7 % (2 à 15 %) |
| &nbsp;&nbsp;dont réelles bien classées | 100 % | 100 % |

Les métriques de génération sont calculées sur 200 formes générées contre 200 réelles (40 par famille). Un 1-NNA global proche de 50 % ne veut **pas** dire que les formes générées sont indiscernables : il se décompose en deux moitiés très inégales. Une vraie forme a toujours pour plus proche voisin une autre vraie forme (100 %), car les formes générées, plus diffuses, restent plus loin d'elle que ses voisines réelles. À l'inverse, une forme générée a presque toujours pour plus proche voisin une vraie forme (7 %) : elle se rapproche des vraies plus que des autres formes générées. Le score global n'est donc qu'une moyenne de ces deux erreurs opposées ; c'est la séparation des deux moitiés qui mesure l'écart restant, et `evaluate_generation.py` l'affiche. L'EMA devient aussi nettement utile : sans elle, l'écart médian remonte à 0.006–0.008. Tirer 30 % des instants d'entraînement dans $[0.9, 1]$ (`late_t_prob`), là où se résout le détail de surface, aide à 30 époques (MMD 0.00131 contre 0.00159) mais n'apporte presque plus rien à 90 (MMD 0.00093, un run) ; l'option reste désactivée.

### Nombre de pas d'intégration (couplage indépendant)

Sur le modèle à 90 époques (seed 0), avant le couplage OT, en faisant varier `sampling.num_steps` et `sampling.solver` :

| Solveur, pas | MMD-CD | COV-CD | Écart médian à la surface |
|---|---|---|---|
| Heun 2 | 0.00297 | 23 % | 0.0147 |
| Heun 3 | 0.00176 | 40 % | 0.0108 |
| Heun 5 | 0.00125 | 46 % | 0.0073 |
| **Heun 10** | 0.00108 | 57 % | 0.0059 |
| **Heun 20** | 0.00101 | 56 % | 0.0053 |
| Heun 40 | 0.00098 | 56 % | 0.0047 |
| Euler 10 | 0.00161 | 18 % | 0.0097 |
| Euler 20 | 0.00110 | 35 % | 0.0067 |
| Euler 40 | 0.00100 | 46 % | 0.0049 |

Chaque pas de Heun coûte deux évaluations du réseau. Heun 10 (20 évaluations) garde presque toute la qualité pour deux fois moins de calcul que le défaut, et fait mieux qu'Euler 20 à coût égal, surtout en couverture. En dessous de 5 pas, la qualité s'effondre. Le couplage OT ci-dessous change ce tableau.

### Couplage OT par forme

Après 90 époques, la surface plafonnait autour de 0.005 d'écart médian, et aucun des leviers suivants ne la faisait vraiment descendre (un run chacun, 200 contre 200, référence 0.0050 à 0.0053) :

| Essai | Écart médian à la surface | MMD-CD |
|---|---|---|
| Taux d'apprentissage en cosinus avec échauffement (`lr_schedule`) | 0.0052 | 0.00102 |
| Modèle élargi (largeur 384, 23 M de paramètres) | 0.0049 | 0.00094 |
| Temps multiplié par 1000 avant l'encodage sinusoïdal (`time_scale`, 2 runs) | 0.0050 | 0.00095 à 0.00096 |
| Attention locale sur les 8 plus proches voisins dans chaque bloc (`local_k`) | 0.0054 | 0.00106 |

Le plateau n'est pas non plus un défaut de généralisation (même écart, 0.0053, sur des squelettes d'entraînement et sur des squelettes nouveaux) ni une ambiguïté d'échelle (l'écart reste entre 0.005 et 0.008 pour des rayons d'os de 0.04 à 0.32, alors qu'une erreur d'échelle le rendrait proportionnel au rayon). Un flou de largeur constante, qui résiste à la capacité, à la durée et à l'optimiseur, désigne plutôt la cible d'apprentissage : avec un couplage indépendant, le signal du détail de surface est noyé dans la variance de $x_1 - x_0$.

| 90 époques, bf16 | Couplage indépendant | OT + 20 % de bruit indépendant | **OT pur** (2 runs) |
|---|---|---|---|
| Loss finale | 1.25 | 0.29 | **0.013** |
| Écart médian à la surface | 0.0052 | 0.0094 | **0.0040 / 0.0042** |
| Biais vers l'extérieur (écart signé moyen) | +0.0019 | +0.0033 | **+0.0003 / +0.0002** |
| MMD-CD | 0.00097 | 0.00162 | **0.00092 / 0.00095** |
| COV-CD | 55 % | 50 % | 57 % / 52 % |
| Famille reconnue | 100 % | 100 % | **100 % / 100 %** |
| CD vers sa famille | 0.0010 | 0.0018 | **0.0010 / 0.0010** |

L'appariement pur réduit l'écart à la surface d'environ 20 % et fait presque disparaître le biais vers l'extérieur. Mélanger du bruit indépendant, même 20 %, ramène la variance et fait pire que la référence. Avec le générateur v2, le couplage OT faisait chuter la reconnaissance de famille à 42 % ; avec le générateur v4, il n'y a plus de perte.

Les trajectoires deviennent presque droites, ce qui rend l'échantillonnage bien moins coûteux (run 0) :

| Solveur, pas | Évaluations du réseau | Écart médian à la surface | MMD-CD | COV-CD |
|---|---|---|---|---|
| Heun 2 | 4 | 0.0071 | 0.00124 | 57 % |
| Heun 3 | 6 | 0.0058 | 0.00107 | 56 % |
| Heun 5 | 10 | 0.0048 | 0.00098 | 57 % |
| Euler 5 | 5 | 0.0039 | 0.00093 | 57 % |
| **Euler 10** (défaut) | 10 | **0.0038** | **0.00092** | 58 % |
| Heun 10 | 20 | 0.0042 | 0.00094 | 57 % |
| Heun 20 | 40 | 0.0040 | 0.00092 | 57 % |

Euler 5 fait aussi bien que Heun 20 avec huit fois moins d'évaluations ; en Euler, le 1-NNA côté générées remonte un peu (7 à 11 %, contre 0 à 3 % en Heun).

### Perturbation du bruit apparié (Not-So-OT)

Not-So-OT (ICLR 2025) trouve, sur ShapeNet, que l'OT pur fait moins bien qu'un OT légèrement perturbé, $x_0' = \sqrt{1-\beta}\,x_0 + \sqrt{\beta}\,\varepsilon$ (`data.ot_noise_beta`). Ce n'est pas le même mélange que les « 20 % de bruit indépendant » plus haut, qui remplaçaient des tirages entiers. Sur nos créatures (200 contre 200, Euler 10, un run par β sauf mention) :

| | β = 0 (2 runs) | β = 0.05 | β = 0.2 |
|---|---|---|---|
| Écart médian à la surface, 90 époques | 0.0038 / 0.0040 | 0.0036 | 0.0038 |
| MMD-CD, 90 époques | 0.00092 / 0.00094 | 0.00089 | 0.00089 |
| COV-CD, 90 époques | 58 % / 56 % | 54 % | 51 % |
| Écart médian à la surface, 180 époques | 0.0029 / 0.0030 | 0.0029 | |
| MMD-CD, 180 époques | 0.00084 / 0.00085 | 0.00083 | |

L'avance de β = 0.05 à 90 époques est de l'ordre de l'écart entre deux runs de β = 0, et elle disparaît à 180 époques. La perturbation n'apporte rien de mesurable ici, et la couverture baisse quand β augmente : β reste à 0. La différence avec Not-So-OT tient sans doute aux données (des formes simples autour d'un squelette, conditionnées, contre ShapeNet sans condition) et à la métrique.

### Durée d'entraînement avec le couplage OT

Avec la cible devenue presque déterministe, la loss continue de baisser bien après 90 époques (0.0129 à 90, 0.0103 à 360), et la surface s'affine avec elle (200 contre 200, Euler 10) :

| | 90 époques | **180 époques** (2 runs) | 180 époques, cosinus | 360 époques |
|---|---|---|---|---|
| Écart médian à la surface | 0.0038 | **0.0029 / 0.0030** | 0.0030 | 0.0026 |
| p90 | 0.0116 | **0.0088 / 0.0091** | 0.0093 | 0.0076 |
| MMD-CD | 0.00092 | **0.00084 / 0.00085** | 0.00086 | 0.00082 |
| COV-CD | 58 % | 57 % / 56 % | 54 % | 56 % |
| 1-NNA côté générées | 7 % | 2 % / 2 % | 6 % | 1 % |
| Famille reconnue (CD) | 100 % (0.0010) | **100 % (0.0009) / 100 % (0.0009)** | | |

Le gain vient de la durée, pas du calendrier : un taux constant fait aussi bien qu'une décroissance en cosinus. Le run de 360 époques passe par 0.0031 à 180 époques et 0.0028 à 270 : le gain continue mais ralentit, et 180 époques est le défaut (environ 32 min). Réduire les tirages de bruit OT de 8 à 2 par forme ne change rien à 90 époques (écart médian 0.0041 contre 0.0040, MMD 0.00094 contre 0.00092) : la diversité du bruit apparié ne limite pas le modèle.

### OT sur GPU pendant l'entraînement

L'appariement exact (hongrois sur CPU) prend des secondes par forme et ne peut pas être précalculé pour des dizaines de milliers de créatures déformées. `models/ot_gpu.py` le remplace par un Sinkhorn entropique sur GPU (ε recuit de 0.3 à 0.01, 20 itérations), arrondi en permutation exacte, sur 4 blocs aléatoires de 512 points : 0.29 s par lot de 32 à 2048 points, pour un coût de transport 1,09 fois l'optimum (`data.online_ot`). Sur les quadrupèdes (200 contre 200, Euler 10) :

| | OT exact | OT GPU |
|---|---|---|
| Écart médian à la surface, 90 époques | 0.0038 / 0.0040 | 0.0036 |
| Écart médian à la surface, 180 époques | 0.0029 / 0.0030 | 0.0028 |
| MMD-CD, 180 époques | 0.00084 / 0.00085 | 0.00082 |
| COV-CD, 180 époques | 57 % / 56 % | 54 % |

L'approximation ne coûte rien de mesurable : c'est le couplage du modèle à squelette libre.

Le modèle publié et la visionneuse utilisent le générateur v4, `data_scale: 4`, le couplage OT pur, 180 époques en bf16 à taux constant (run 0) et Euler 10.

## 🧬 Données Articulation-XL (en cours)

Étape suivante : passer des quadrupèdes procéduraux à des créatures de toutes espèces (conception : [conception.md](conception.md)). `scripts/build_axl_cache.py` lit [Articulation-XL2.0](https://huggingface.co/datasets/Seed3D/Articulation-XL2.0) (CC-BY-4.0, via [son miroir Parquet](https://huggingface.co/datasets/chibifire/datasource-articulation-xl2-parquet)) groupe de lignes par groupe de lignes, sans télécharger les 25 Go de maillages. Il garde les créatures (animaux, créatures mythiques, personnages, jouets), échantillonne 8192 points avec leurs poids de peau, oriente chaque modèle (haut +Y, avant +X) et le ramène au rayon 1. `data/axl/dataset.py` déforme ensuite chaque créature par son squelette : longueur des os, épaisseur, pose, parties qui rétrécissent ou disparaissent. Il tire aussi 70 % d'animaux.

| | Entraînement | Test |
|---|---|---|
| Créatures retenues | 40 218 (10 948 animaux) | 1 736 (494 animaux) |
| Taille du cache | 4,8 Go | 0,2 Go |
| Orientation, échecs nets à l'œil (200 créatures) | 6 % (12,5 % au pire) | |

Rapport complet et planches : [`axl/report.md`](axl/report.md). Les données dérivées ne sont pas republiées.

### Premier modèle à squelette libre (axl_v3)

10,8 M de paramètres, 10 époques sur 40 209 créatures en bf16, environ 4 h 15 sur une RTX 4060 Laptop. Évaluation sur 200 créatures de test (squelettes jamais vus), Euler 10 :

| | Squelette seul | Squelette + z |
|---|---|---|
| MMD-CD | 0.0125 | 0.0123 |
| COV-CD | 72 % | 73 % |
| 1-NNA (50 % idéal) | 26,5 % | 26,2 % |
| Écart médian au nuage réel | 0.051 | 0.051 |

Sensibilité au squelette (z fixé) : 0.68. Planche : [`axl/plate_axl_v3.png`](axl/plate_axl_v3.png).

Les silhouettes suivent le squelette (posture, proportions, humanoïdes nets), mais la surface est floue, les parties fines (pattes, queues, oreilles) se perdent dans le volume, et les têtes ne sont pas lisibles. Le code de forme z ne sert presque pas : avec le z d'une créature réelle, la génération n'est pas plus proche d'elle (0.0510 contre 0.0512), c'est un effondrement du postérieur (|μ| ≈ 0.07).

Deux premiers essais se sont effondrés à l'époque 6 (fm de 0.075 à 0.58). La cause : le vecteur de conditionnement de l'AdaLN, non normalisé, s'est emballé (norme du plongement de t de 22 à 197, jeton global du squelette de 1,9 à 52, modulations jusqu'à 78). Un LayerNorm sur ce vecteur l'a supprimé. En chemin : 9 rigs dont une articulation est à plus de 1,5 du centre (jusqu'à 78, alors que le maillage tient dans la sphère unité) sont écartés, car l'augmentation projetait leurs points aussi loin et multipliait la perte du lot par 100 ; les pas non finis sont sautés ; le taux d'apprentissage suit un cosinus avec échauffement.

---

## ⚠️ Limites

- **Données procédurales, pas SMAL** : les formes sont des cylindres échantillonnés autour d'un squelette paramétrique inspiré de la topologie SMAL, pas le modèle SMAL lui-même (`shape_betas` est inutilisé). La variabilité intra-famille est faible (±3 % sur les proportions).
- **100 % de reconnaissance, sur un test facile** : avec le générateur v4, les 5 familles sont séparées par construction (proportions, écartement, port de tête) et leur variabilité interne est faible ; ce score mesure que le modèle suit le squelette, pas qu'il généralise à de vrais animaux.
- **Précision** : les formes générées restent environ 1,3 fois plus éloignées de leur famille que deux créatures réelles entre elles (0.0009 contre 0.0007), et leur surface reste plus diffuse que celle des vraies formes (écart médian 0.0029 contre 0.0004). Un classifieur au plus proche voisin reconnaît encore toutes les vraies formes (1-NNA côté réelles : 100 %).
- **L'autoguidage n'aide pas ici** (`--guide-checkpoint`, avec un petit modèle de 8 époques comme guide) : à $w = 1.5$, il porte une seed à 100 % de reconnaissance (modèles du générateur v3), mais divise presque la couverture par deux (COV 62 % → 32 %) ; au-delà, les formes se dégradent. Restreindre le guidage à un intervalle de $t$ (`--guidance-interval`) ne fait que l'atténuer.
- **Greffe** : la jonction aligne le sous-arbre donneur par translation, sans mise à l'échelle ni rotation.
- **Une seule seed par variante intermédiaire** dans l'ablation. Le modèle final varie de 90 à 95 % selon la seed (3 runs), et l'évaluation sur 40 squelettes a elle-même une incertitude d'environ ±8 points.
- **Coût du couplage OT** : l'appariement exact coûte environ 2 s par tirage et par forme ; il est précalculé une fois pour 8 tirages par forme (environ 30 min sur 12 cœurs pour 500 formes). Doubler le jeu de données doublerait ce coût ; 2 tirages par forme donnent toutefois le même résultat que 8.

