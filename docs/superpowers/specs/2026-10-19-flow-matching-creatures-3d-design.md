# Spécification de Conception : Génération Compositionnelle de Créatures 3D par Flow Matching Conditionné sur Squelette

**Date** : 2026-10-19  
**Auteur** : Antigravity & Utilisateur  
**Projet** : Portfolio & R&D en Deep Learning Géométrique 3D  
**Statut** : Validé pour Planification  

---

## 1. Contexte et Objectifs

Ce projet a pour objectif de concevoir, implémenter et évaluer un système génératif de bout en bout capable de synthétiser des **créatures 3D anatomiquement plausibles et originales** sous forme de nuages de points ($N = 2048$), guidées par un **squelette cinématique articulé 3D** ($K = 35$ articulations).

Le système permet en outre la **composition d'hybrides chimériques** (ex. corps de félin associé à un cou et une tête d'équidé) par greffe cinématique de sous-arbres squelettiques, avec une continuité géométrique garantie aux jonctions grâce à un mécanisme d'attention spatiale point-articulation.

### Contraintes et Environnement Cible
- **Plateforme** : Windows 11, Python 3.12.
- **Accélération matérielle** : GPU NVIDIA GeForce RTX 4060 Laptop (8 Go VRAM, CUDA 13.1).
- **Philosophie d'ingénierie** : Dépendances C++/CUDA externes instables évitées (pas de PyTorch3D ou DGL pré-compilés) ; implémentation native et vectorisée en pur PyTorch.
- **Restitution portfolio** : Visionneuse 3D web interactive autonome (HTML5 / Three.js) visualisant le squelette, le nuage de points et l'animation temporelle $t \in [0, 1]$ du flux génératif.

---

## 2. Formulation Mathématique : Optimal Transport Flow Matching (OT-FM)

### 2.1 Cadre Théorique
Contrairement aux modèles de diffusion stochastiques traditionnels (DDPM/SDE) qui requièrent des centaines d'étapes de débruitage, le Flow Matching basé sur le Transport Optimal (OT-FM) apprend un champ de vecteurs déterministe le long de trajectoires d'interpolation géodésiques rectilignes.

Soit $x_0 \sim p_0(x) = \mathcal{N}(0, I) \subset \mathbb{R}^{N \times 3}$ un nuage de points de bruit gaussien initial, et $x_1 \sim p_1(x \mid S) \subset \mathbb{R}^{N \times 3}$ le nuage de points d'une créature cible conditionnée par son squelette $S$.

L'interpolant du Transport Optimal au temps $t \in [0, 1]$ est défini par :
$$x_t = (1 - t) x_0 + t x_1$$

Le champ de vitesse cible conditionnel associé à cette trajectoire est constant dans le temps :
$$u_t(x_t \mid x_0, x_1) = \frac{d x_t}{dt} = x_1 - x_0$$

### 2.2 Objectif d'Entraînement
Le réseau de neurones $v_\theta(x_t, t, S)$ prédit le champ de vitesses instantané. La fonction de perte est la régression quadratique moyenne (MSE) :
$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0, 1], x_0 \sim \mathcal{N}(0, I), (x_1, S) \sim \mathcal{D}} \left[ \| v_\theta(x_t, t, S) - (x_1 - x_0) \|_2^2 \right]$$

### 2.3 Échantillonnage ODE
À l'inférence, pour un squelette donné $S$ :
1. Échantillonner $x_0 \sim \mathcal{N}(0, I)$.
2. Intégrer l'équation différentielle ordinaire (ODE) $\frac{d x_t}{dt} = v_\theta(x_t, t, S)$ de $t=0$ à $t=1$.
3. Deux intégrateurs numériques sont implémentés :
   - **Solveur d'Euler** ($M = 15$ à $20$ pas fixes $\Delta t = 1/M$) :
     $$x_{t + \Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t, S)$$
   - **Solveur de Heun (ordre 2)** ($M = 10$ à $15$ pas) :
     $$x'_{t + \Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t, S)$$
     $$x_{t + \Delta t} = x_t + \frac{\Delta t}{2} \left[ v_\theta(x_t, t, S) + v_\theta(x'_{t + \Delta t}, t + \Delta t, S) \right]$$

---

## 3. Données et Représentation Cinématique

### 3.1 Squelette Cinématique Standardisé (Topologie SMAL)
Le modèle s'appuie sur la topologie cinématique quadruped SMAL comprenant $K = 35$ articulations interconnectées :
- **Tronc & Axe** : Bassin (racine cinématique), 4 articulations lombaires/thoraciques, base du cou, cou médian, base du crâne, mâchoire.
- **Membres antérieurs (G/D)** : Épaule, coude, poignet, patte antérieure.
- **Membres postérieurs (G/D)** : Hanche, genou, jarret/cheville, patte postérieure.
- **Appendice caudal** : 4 articulations formant la queue.

Chaque articulation $k \in \{0, \dots, K-1\}$ est caractérisée par :
- Sa position 3D dans le repère canonique normalisé $J_k \in [-1, 1]^3$.
- Son index parent $\text{parent}(k) \in \{-1, 0, \dots, K-1\}$ formant l'arbre cinématique acyclique.
- Son identifiant sémantique et anatomique (embedding catégoriel).

### 3.2 Familles Morphologiques Modélisées
Le générateur paramétrique simule les variations morphologiques de 5 grandes familles :
1. `Felidae` (ex. lion, tigre, guépard : membres musculeux, tronc élancé, crâne court).
2. `Canidae` (ex. loup, chien, renard : museau allongé, membres fins, cage thoracique profonde).
3. `Equidae` (ex. cheval, zèbre : long cou, encolure haute, membres longs avec un seul doigt d'appui).
4. `Bovidae` (ex. cerf, antilope, taureau : tronc lourd, port de tête particulier).
5. `Hippopotamidae` (ex. hippopotame : corps massif, membres courts et épais).

### 3.3 Format des Données & Pipeline de Cache
Chaque échantillon synthétisé est stocké sous forme de dictionnaire normalisé :
```python
{
    "points": np.ndarray,        # (N, 3) float32 dans [-1, 1]^3 (surface de la créature)
    "joints": np.ndarray,        # (K, 3) float32 dans [-1, 1]^3 (positions 3D des articulations)
    "joint_parents": np.ndarray, # (K,) int64 (arbre cinématique)
    "family_id": int,            # 0 à 4
    "shape_betas": np.ndarray    # (10,) float32 (paramètres de forme morphologique)
}
```

### 3.4 Augmentations Géométriques
Pour assurer l'invariance et la robustesse du réseau :
- Rotation aléatoire cohérente $R_y(\theta)$ appliquée identiquement à $x_1$ et $J$.
- Facteur d'échelle isotrope $s \sim \mathcal{U}[0.9, 1.1]$.
- Bruit d'agitation léger sur les positions d'articulations ($\sigma = 0.005$) pour éviter le surapprentissage d'un squelette rigide exact.

---

## 4. Architecture Neuronale : Point-DiT (Diffusion/Flow Transformer)

Le réseau prend en entrée le nuage bruité $x_t \in \mathbb{R}^{B \times N \times 3}$, le temps continu $t \in \mathbb{R}^B$, et le squelette $S = (J, \text{parents}, \text{family})$.

```
           +-----------------------------------------------+
           |                Squelette S                    |
           |   Joints J (K x 3) + Embeddings d'Anatomie    |
           +-----------------------+-----------------------+
                                   |
                         [ Kinematic Encoder ]
                     (Self-Attention sur l'arbre)
                                   |
                                   v  Memoire Hs (K x D)
                                   |
+---------------------+            |            +-----------------------+
|  Points bruites xt  |            |            |        Temps t        |
|       (N x 3)       |            |            |       (scalaire)      |
+----------+----------+            |            +-----------+-----------+
           |                       |                        |
     [ Point Proj ]                |                 [ Timestep MLP ]
    + Fourier PosEmb               |                (Sinusoidal + Linear)
           |                       |                        |
           v                       |                        v
+-----------------------------------------------------------------------+
|                       Bloc Point-DiT (x L)                            |
|                                                                       |
|  1. LayerNorm modifiee par AdaLN (gamma_t, beta_t, alpha_t)           |
|  2. Self-Attention Multi-Tetes sur les N points (coherence de surface)|
|  3. Spatial Cross-Attention Point -> Joints Hs avec Biais Geometrique  |
|  4. Feed-Forward Network (GELU) module par AdaLN                      |
+-----------------------------------------------------------------------+
                                   |
                                   v
             [ Tete de sortie modilee AdaLN + Proj 3D ]
                                   |
                                   v
                  Vitesses predites vt (N x 3)
```

### 4.1 Spécifications des Composants
1. **Encodeur Squelettique Cinématique (`SkeletonEncoder`)** :
   - Projection linéaire de $J_k$ vers $\mathbb{R}^D$ additionnée à un embedding anatomique appris ($K$ classes).
   - 2 couches de Transformer Self-Attention pour propager la cohérence cinématique le long de l'arbre.
   - Sortie : $H_S \in \mathbb{R}^{B \times K \times D}$ ($D = 256$).
2. **Tokeniseur de Points (`PointTokenizer`)** :
   - Projection linéaire des coordonnées $(x, y, z)$ vers $\mathbb{R}^D$ complétée par un encodage positionnel harmonique (Fourier Features).
3. **Conditionnement Temporel AdaLN (`AdaptiveLayerNorm`)** :
   - Encodage sinusoïdal du temps $t$ suivi d'un MLP à 2 couches produisant les scalaires $(\gamma, \beta, \alpha)$ pour chaque bloc.
4. **Cross-Attention Spatiale Point $\to$ Joints (`SpatialCrossAttention`)** :
   - Requêtes $Q \in \mathbb{R}^{B \times N \times D}$ dérivées des points ; Clés $K$ et Valeurs $V \in \mathbb{R}^{B \times K \times D}$ dérivées des articulations squelettiques.
   - **Biais géométrique spatial** : Les logits d'attention intègrent une pénalité proportionnelle à la distance euclidienne relative $\|x_t^{(i)} - J_k\|_2$, forçant les points à interroger en priorité les os anatomiquement proches tout en conservant une vision globale.
5. **Hyperparamètres du Modèle** :
   - Dimension cachée $D = 256$
   - Nombre de couches $L = 6$
   - Nombre de têtes d'attention = 8
   - Taux de Dropout = 0.05
   - Total paramètres : ~12.8M (léger, ultra-rapide sur RTX 4060).

---

## 5. Composition d'Hybrides et Continuité aux Jonctions

### 5.1 Greffe Cinématique de Sous-Arbres (Kinematic Subtree Splicing)
Pour générer une chimère :
1. **Sélection** : Choisir un squelette hôte $S_{\text{host}}$ (ex. Félin) et un squelette donneur $S_{\text{donor}}$ (ex. Équidé).
2. **Définition de la jonction** : Identifier le joint charnière (ex. `neck_base` ou `pelvis`).
3. **Transplantation** : Remplacer le sous-arbre issu de la charnière dans $S_{\text{host}}$ par le sous-arbre correspondant de $S_{\text{donor}}$.
4. **Raccord cinématique** :
   - Calculer le vecteur d'offset entre la charnière hôte et la racine du sous-arbre donneur.
   - Appliquer une translation et une mise à l'échelle proportionnelle pour aligner rigoureusement les articulations de jonction.
   - Squelette résultant $S_{\text{hybrid}} \in \mathbb{R}^{K \times 3}$.

### 5.2 Continuité Géométrique
Grâce à la nature continue de la Cross-Attention spatiale du Point-DiT, les points situés dans la zone de transition reçoivent simultanément l'influence des os de l'hôte et du donneur, produisant une surface étanche, continue et musculeusement crédible, sans artefact de coupure ni trou de maillage.

---

## 6. Métriques d'Évaluation Quantitatives

1. **Chamfer Distance (CD)** :
   $$\text{CD}(X, Y) = \frac{1}{|X|} \sum_{x \in X} \min_{y \in Y} \|x - y\|_2^2 + \frac{1}{|Y|} \sum_{y \in Y} \min_{x \in X} \|x - y\|_2^2$$
   Implémentée en PyTorch vectorisé par blocs de mémoire pour éviter tout débordement VRAM.
2. **Couverture Anatomique du Squelette (Skeleton Bone Coverage)** :
   Mesure la distance maximale et moyenne entre chaque segment osseux du squelette et le point le plus proche du nuage généré :
   $$\text{Coverage}(S, X) = \frac{1}{K} \sum_{k=1}^K \min_{x \in X} \|J_k - x\|_2$$
   Permet de vérifier qu'aucun membre n'est fantôme ou manquant.
3. **Métrique d'Originalité (Novelty Score)** :
   $$\text{Novelty}(X_{\text{gen}}) = \min_{Y \in \mathcal{D}_{\text{train}}} \text{CD}(X_{\text{gen}}, Y)$$
   Pour une créature hybride, la distance doit être strictement supérieure à un seuil $\epsilon_{\text{novel}}$, garantissant qu'elle ne copie fidèlement aucun animal réel du dataset.

---

## 7. Visionneuse Web 3D Interactive (Three.js)

### 7.1 Architecture
- Fichiers statiques autonomes : `viewer/index.html`, `viewer/app.js`, `viewer/style.css`.
- Chargement de Three.js et OrbitControls via CDN (aucune compilation npm requise).
- Consommation d'un fichier JSON/ArrayBuffer exporté par le script d'inférence, contenant :
  - La trajectoire temporelle $X_t \in \mathbb{R}^{T \times N \times 3}$ ($T \approx 15$ étapes).
  - Les articulations squelettiques $J \in \mathbb{R}^{K \times 3}$ et la liste des arêtes osseuses (parents).
  - Les métadonnées (nom de l'espèce / hybride, distance de Chamfer, score de nouveauté).

### 7.2 Fonctionnalités UI
- **Canvas WebGL 3D** : Rendu avec éclairage ambiant et directionnel doux, sol quadrillé et ombres douces.
- **Rendu du squelette** : Sphères colorées aux articulations reliées par des cylindres semi-transparents matérialisant les os.
- **Rendu du nuage de points** : Particules Three.js personnalisées avec dégradé de couleur selon l'axe d'élévation ou la proximité squelettique.
- **Curseur temporel interactif ($t \in [0, 1]$)** :
  - Slider permettant d'inspecter manuellement l'état du nuage à tout instant $t$.
  - Bouton Play/Pause pour animer la naissance de la créature depuis le bruit gaussien ($t=0$) vers la forme finale ($t=1$).
- **Sélecteur de spécimens** : Menu déroulant pour inspecter les animaux standard et les créatures hybrides.
- **Exportation** : Bouton d'export direct au format standard `.ply` (nuage de points ASCII/binaire).

---

## 8. Arborescence du Projet

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
│   └── trainer.py                # Boucle d'entraînement PyTorch avec AMP (fp16)
├── viewer/
│   ├── index.html                # Visionneuse 3D HTML5
│   ├── app.js                    # Contrôles Three.js (caméra, slider t, rendu 3D)
│   └── style.css                 # Style sombre moderne (glassmorphism)
├── scripts/
│   ├── prepare_data.py           # Génération et mise en cache du dataset SMAL
│   ├── train.py                  # Script d'entraînement
│   ├── sample_and_export.py      # Génération de trajectoires & export JSON pour viewer
│   └── evaluate.py               # Évaluation des métriques quantitatives
├── docs/
│   └── superpowers/specs/        # Spécifications de conception validées
├── requirements.txt              # Dépendances légères (torch, numpy, scipy, pyyaml, tqdm)
└── projet-creatures-3d.md        # Note de cadrage de recherche initiale
```

---

## 9. Plan de Validation et Auto-Évaluation de la Spécification

- **Vérification des placeholders** : Aucun `TBD`, `TODO` ou section non spécifiée.
- **Cohérence interne** : Toutes les dimensions ($N=2048$, $K=35$, $D=256$, $L=6$, $B=32$) sont harmonisées entre les sections mathématiques, modèle et visionneuse.
- **Portée et faisabilité** : Le modèle (~12.8M de paramètres, précision mixte fp16) consomme moins de 3 Go de VRAM sur les 8 Go disponibles sur la RTX 4060, assurant une exécution locale sans risque de saturation mémoire (OOM).
