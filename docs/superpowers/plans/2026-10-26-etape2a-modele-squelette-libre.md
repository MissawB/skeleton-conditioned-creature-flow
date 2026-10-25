# Étape 2a : modèle à squelette libre, code de forme et OT sur GPU — plan de réalisation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** entraîner sur le cache Articulation-XL un premier modèle qui génère un nuage de 2048 points à partir de n'importe quel squelette (2 à 255 articulations), avec un code de forme $z$ optionnel, en quelques heures sur la RTX 4060, et le mesurer.

**Architecture:**
- L'appariement OT exact précalculé est remplacé par un appariement Sinkhorn sur GPU, calculé pendant l'entraînement. Il est d'abord validé sur le jeu quadrupède.
- Un nouvel encodeur de squelette par graphe (recette AnyTop/UniMate) alimente un `CreatureDiT`, qui reprend les blocs du Point-DiT avec des masques d'articulations et un biais de présence.
- Un encodeur de nuage produit $z$, avec une régularisation KL et un abandon aléatoire.
- L'ancien pipeline quadrupède reste intact.

**Tech Stack:** PyTorch 2.11 (bf16, `scaled_dot_product_attention`), NumPy, le cache `data/cache/axl` et `data/axl/dataset.py` de l'étape 1.

**Spec:** `docs/superpowers/specs/2026-10-25-creatures-toutes-especes-chimeres-design.md`, étape 2 (§5). L'état de l'art est dans `reports/État des recherches chimères 3D.md`, parties 3 et 4.

**Hors de ce plan (plan 2b) :** la passe « visage » dense, l'export ONNX et le microbenchmark navigateur.

## Global Constraints

- Python : `.venv/Scripts/python`. Jamais le Python système. Pur PyTorch, sans extension compilée (ni `torch_cluster`, ni KeOps, ni POT).
- Tests sans GPU : `CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python -m pytest -q`. Les 105 tests existants restent verts, et `PointDiT` ainsi que `OptimalTransportFlowMatching` gardent leur comportement par défaut.
- Un seul job GPU à la fois. Lancer les entraînements avec `SetThreadExecutionState(0x80000001)` et prévenir l'utilisateur que la veille doit être désactivée.
- Choix figés par l'étape 1 et la spec : `ot_noise_beta = 0`, bf16, EMA 0.995, génération Euler 10 pas, équilibre 70/30 animaux/humanoïdes (`sampling_weights`), modèle de 10 à 30 M paramètres.
- Seuil de validation de l'OT sur GPU (spec) : écart médian à la surface ≤ 0.0045 sur le jeu quadrupède à 90 époques. L'appariement exact donne 0.0040, l'absence d'OT 0.0052. Confirmation à 180 époques contre 0.0029/0.0030.
- Disque : environ 38 Go libres. Sauvegarder un checkpoint toutes les 10 époques au plus, et le dernier.
- Branche : `feat/stage2a-free-skeleton`. Commits terminés par `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- Un lot qui mélange un squelette à 2 articulations et un autre à 255 : les articulations de remplissage ne doivent rien changer à la sortie des vraies. Test : Task 2, `test_padding_does_not_change_real_joint_outputs`, et Task 3, `test_padded_joints_do_not_change_the_velocity`.
- Une articulation de présence 0 : le biais $\log w$ vaut $-\infty$, ce qui ne doit produire ni NaN, ni une ligne d'attention entièrement masquée. Test : Task 3, `test_zero_presence_is_finite_and_ignored`.
- Un DataLoader avec des workers sous Windows (spawn), alors que le processus principal a déjà lu des exemples : les memmaps ne doivent pas être copiées dans chaque worker. Test : Task 5, `test_dataset_pickles_without_open_memmaps`.
- La génération sans $z$ (`z=None`, squelette seul) et la génération avec un $z$ nul doivent donner le même résultat. Test : Task 3, `test_missing_z_equals_the_null_code`.
- Un appariement Sinkhorn qui renvoie une non-permutation (doublons après l'arrondi) : le résultat doit toujours être une permutation exacte. Test : Task 1, `test_sinkhorn_returns_a_permutation`.

---

### Task 1 : appariement OT par forme sur GPU, et validation sur le jeu quadrupède

**Files:**
- Create: `models/ot_gpu.py`, `tests/test_ot_gpu.py`
- Modify: `models/flow_matching.py` (`OptimalTransportFlowMatching.__init__`, `compute_loss`), `scripts/train.py`, `configs/default_config.yaml`

**Interfaces:**
- Produces:
  - `sinkhorn_permutation(x0: Tensor (B,N,3), x1: Tensor (B,N,3), eps: float = 0.01, iters: int = 50) -> LongTensor (B,N)` : une permutation `perm` telle que `x0[b, perm[b, j]]` est le partenaire de `x1[b, j]`.
  - `ot_match(x0, x1, **kw) -> Tensor` : renvoie `x0` réordonné.
  - `OptimalTransportFlowMatching(..., online_ot: bool = False)` : quand `x_0` n'est pas fourni et que `online_ot` est vrai, le bruit tiré est apparié par `ot_match` dans l'espace mis à l'échelle (`x_1 * data_scale`).
  - Clé de config `data.online_ot: false`.

Algorithme :
- Coût : distance euclidienne au carré, normalisée par sa moyenne.
- Plan : Sinkhorn en domaine log, `eps` fixe, `iters` itérations, marges uniformes.
- Arrondi en permutation, par tours vectorisés : chaque point de données libre propose le bruit libre de plus forte valeur du plan. En cas de conflit, le bruit va à la plus forte proposition. On répète jusqu'à ce que tout soit apparié. Au-delà de 20 tours, les restes sont appariés dans l'ordre.

- [ ] **Step 1 : écrire les tests qui échouent**

```python
def test_sinkhorn_returns_a_permutation():
    # B=3, N=257 (taille impaire), points aléatoires et nuages dupliqués (x1 avec 10 points identiques)
    # pour chaque b : sorted(perm[b]) == range(N)

def test_sinkhorn_cost_is_close_to_exact_matching():
    # 4 quadrupèdes SMALQuadrupedGenerator à 512 points, mis à l'échelle ×4, bruit N(0, I)
    # coût moyen ||x0[perm]-x1||² ≤ 1.03 × coût de scipy linear_sum_assignment, et ≤ 0.6 × coût sans appariement

def test_flow_matching_online_ot_pairs_noise_with_the_shape():
    # OptimalTransportFlowMatching(model factice qui renvoie zéros, online_ot=True) : la loss calculée est la
    # même qu'avec x_0 = ot_match(bruit, x_1*s) passé explicitement (graine torch fixée)
```

- [ ] **Step 2 : vérifier qu'ils échouent** — `pytest tests/test_ot_gpu.py -v`, échec attendu à l'import.
- [ ] **Step 3 : implémenter** `models/ot_gpu.py`, puis le paramètre `online_ot`. Passer `data.online_ot` depuis `scripts/train.py`. Avec `online_ot: true`, `ot_noise_per_sample` doit valoir 0 : lever une erreur claire si les deux sont actifs.
- [ ] **Step 4 : vérifier** — toute la suite passe. Mesurer sur GPU le temps d'appariement pour B=32, N=2048, à consigner dans le ledger. Cible : moins de 20 % du temps d'un pas d'entraînement. Sinon, réduire `iters` et vérifier de nouveau le test de coût.
- [ ] **Step 5 : commit** — `feat: online per-shape OT matching on the GPU (Sinkhorn + permutation rounding)`
- [ ] **Step 6 : validation sur le jeu quadrupède (GPU, environ 20 min)** — config `ot8_ind0_seed0` avec `ot_noise_per_sample: 0`, `online_ot: true`, 90 époques, Euler 10, puis `scripts/evaluate_generation.py`.
  - **Réussi** si l'écart médian à la surface est ≤ 0.0045.
  - Sinon, deux tentatives au plus, avec `eps` 0.005 puis `iters` 100.
  - En cas d'échec, s'arrêter et présenter les chiffres à l'utilisateur.
  - Si c'est réussi, lancer la confirmation à 180 époques (référence 0.0029/0.0030) en arrière-plan, pendant les tâches suivantes.
  - Tableau dans le README, section « Couplage OT par forme ».
  - Commit : `docs: online GPU OT matching validated on the quadruped set`.

---

### Task 2 : encodeur de squelette par graphe

**Files:**
- Create: `models/skeleton_graph.py`, `tests/test_skeleton_graph.py`

**Interfaces:**
- Produces:
  - `REL_SELF, REL_PARENT, REL_CHILD, REL_SIBLING, REL_OTHER = 0, 1, 2, 3, 4` et `MAX_TREE_DIST = 5`.
  - `tree_relations(parents: LongTensor (B,J), joint_mask: BoolTensor (B,J)) -> tuple[LongTensor (B,J,J), LongTensor (B,J,J)]` :
    - la distance dans l'arbre, plafonnée à `MAX_TREE_DIST`, avec `MAX_TREE_DIST + 1` pour toute paire impliquant une articulation de remplissage ;
    - le type de relation (relation de `i` vers `j`).
  - `tree_depth_batch(parents, joint_mask) -> LongTensor (B,J)`.
  - `SkeletonGraphEncoder(hidden_dim: int = 256, num_layers: int = 2, num_heads: int = 4, max_depth: int = 31)` avec `forward(joints, parents, presence, joint_mask) -> tuple[Tensor (B,J,D), Tensor (B,D)]` : les jetons d'articulation et un jeton global.
    - Entrée par articulation : MLP sur `[position, position - position du parent (0 pour la racine), presence]`, plus un embedding de la profondeur, plafonnée à `max_depth`.
    - Attention : un biais appris par tête, `Embedding(MAX_TREE_DIST + 2, heads)` sur la distance plus `Embedding(5, heads)` sur la relation, ajouté aux scores. Les clés de remplissage sont masquées.
    - Jeton global : moyenne masquée des jetons, puis un MLP.

Les distances dans l'arbre se calculent par profondeur et plus proche ancêtre commun, vectorisés sur le lot. Pour J ≤ 255, une boucle sur les niveaux de profondeur suffit.

- [ ] **Step 1 : écrire les tests qui échouent**

```python
def test_tree_relations_on_a_small_tree():
    # parents [-1, 0, 0, 1] : dist[3,2] == 3, dist[1,2] == 2, rel[1,0] == REL_PARENT (0 est le parent de 1),
    # rel[0,1] == REL_CHILD, rel[1,2] == REL_SIBLING, rel[3,2] == REL_OTHER, dist[i,i] == 0

def test_tree_distance_is_capped():
    # chaîne de 12 articulations : dist[0,11] == MAX_TREE_DIST

def test_encoder_is_equivariant_to_joint_relabelling():
    # permuter les articulations (et réindexer parents) permute les jetons de sortie, le jeton global est inchangé (atol 1e-5)

def test_padding_does_not_change_real_joint_outputs():
    # squelette à 2 articulations seul, puis dans un lot complété jusqu'à 255 : mêmes jetons réels et même global (atol 1e-5)
```

- [ ] **Step 2 : vérifier qu'ils échouent.**
- [ ] **Step 3 : implémenter.** Attention via `F.scaled_dot_product_attention` avec un `attn_mask` flottant (biais plus `-inf` sur les clés de remplissage).
- [ ] **Step 4 : vérifier** — les nouveaux tests et toute la suite passent.
- [ ] **Step 5 : commit** — `feat: free-topology skeleton encoder (tree distance and relation attention bias)`

---

### Task 3 : CreatureDiT (masques, présence, code de forme)

**Files:**
- Modify: `models/point_dit.py` (`SpatialCrossAttention.forward`, `PointDiTBlock.forward` : nouveaux arguments optionnels `key_bias: Optional[Tensor] = None`, de forme `(B,1,1,K)`, ajouté aux scores avant le softmax ; rien ne change quand il vaut None)
- Create: `models/creature_dit.py`, `tests/test_creature_dit.py`

**Interfaces:**
- Consumes: `SkeletonGraphEncoder` (Task 2), `PointDiTBlock`, `TimestepEmbedder`.
- Produces:
  - `CreatureDiT(hidden_dim=256, num_layers=6, num_heads=8, z_dim=64, spatial_distance_scale=2.0/s**2, data_scale=s)`, avec `data_scale` en buffer comme dans `PointDiT`.
  - `forward(x_t (B,N,3), t (B,), joints (B,J,3), parents (B,J), presence (B,J), joint_mask (B,J), z: Optional[Tensor (B,z_dim)] = None, cond_mask: Optional[BoolTensor (B,)] = None) -> Tensor (B,N,3)`.
  - Les articulations sont multipliées par `data_scale` comme dans `PointDiT`, puisque les points arrivent déjà mis à l'échelle.
  - `key_bias = log(clamp(presence, min=1e-4))`, plus `-inf` pour le remplissage.
  - Conditionnement AdaLN des blocs : `t_emb + skeleton_global + z_proj(z)`. `z=None` signifie un code nul, et `z_proj` est initialisé à zéro.
  - `cond_mask` False (abandon du squelette) remplace les jetons d'articulation par un jeton nul appris et coupe le biais de distance, comme `PointDiT`.
  - Garder l'estimation $\hat{x}_1$ par bloc (`pos_heads`) de `PointDiT`.

- [ ] **Step 1 : écrire les tests qui échouent**

```python
def test_padded_joints_do_not_change_the_velocity():
    # même créature (J=5) seule et dans un lot complété à J=40 : vitesses identiques (atol 1e-5)

def test_zero_presence_is_finite_and_ignored():
    # presence[:, 3] = 0 : sortie finie ; changer joints[:, 3] ne change plus la vitesse (atol 1e-5)

def test_missing_z_equals_the_null_code():
    # forward(z=None) == forward(z=zeros) ; après un pas d'optimiseur sur z_proj, z non nul change la sortie

def test_output_shape_and_old_point_dit_unchanged():
    # sortie (B,N,3) ; PointDiT existant, mêmes poids, key_bias None : sortie identique à avant (comparaison à un appel sans le nouvel argument)
```

- [ ] **Step 2 : vérifier qu'ils échouent.**
- [ ] **Step 3 : implémenter.**
- [ ] **Step 4 : vérifier** — les nouveaux tests et toute la suite passent, dont les anciens tests de `PointDiT`.
- [ ] **Step 5 : commit** — `feat: CreatureDiT with masked free skeletons, presence bias and shape code`

---

### Task 4 : encodeur de forme $z$

**Files:**
- Create: `models/shape_encoder.py`, `tests/test_shape_encoder.py`

**Interfaces:**
- Produces:
  - `ShapeEncoder(z_dim: int = 64, hidden_dim: int = 128, num_points: int = 512)` avec `forward(points (B,N,3)) -> tuple[Tensor (B,z_dim), Tensor (B,z_dim)]` (mu, logvar). Points mis à l'échelle comme ceux du flow, sous-échantillonnés aléatoirement à `num_points` s'ils sont plus nombreux. MLP partagé par point (3 → 128 → 256), puis concaténation des poolings max et moyenne, puis un MLP vers `2·z_dim`.
  - `reparameterize(mu, logvar) -> Tensor`.
  - `kl_divergence(mu, logvar) -> Tensor` : scalaire, moyenne sur le lot, somme sur les dimensions.

- [ ] **Step 1 : écrire les tests qui échouent** :
  - `test_encoding_is_invariant_to_point_order` : avec `num_points >= N`, aucun sous-échantillonnage.
  - `test_kl_is_zero_for_the_prior` : mu = 0, logvar = 0 donne 0.
  - `test_reparameterize_has_the_right_moments` : 20 000 tirages, moyenne et écart-type à 0.02 près.
- [ ] **Step 2 à 4 :** vérifier qu'ils échouent, implémenter, puis faire passer toute la suite.
- [ ] **Step 5 : commit** — `feat: point cloud shape encoder for the creature code z`

---

### Task 5 : flow des créatures et script d'entraînement

**Files:**
- Create: `models/creature_flow.py`, `scripts/train_axl.py`, `configs/axl_config.yaml`, `tests/test_creature_flow.py`
- Modify: `pipeline/trainer.py` (si le flow expose `loss_from_batch(batch) -> Tensor`, `train_epoch` l'appelle à la place de `compute_loss(pts, jts, x_0)`), `data/axl/dataset.py` (`__getstate__` qui vide `_arrays`)

**Interfaces:**
- Consumes: Tasks 1 à 4, `AXLDataset`, `collate_skeletons`, `sampling_weights` (étape 1).
- Produces:
  - `CreatureFlowMatching(model: CreatureDiT, shape_encoder: ShapeEncoder, kl_weight: float = 1e-3, z_drop_prob: float = 0.3, cond_drop_prob: float = 0.0, ot_iters: int = 50)`.
  - `loss_from_batch(batch: dict) -> Tensor` : FM à chemins linéaires avec appariement `ot_match` en ligne, `t ~ U[0,1]`. $z$ est tiré de `ShapeEncoder(x_1)` par reparamétrage, puis remplacé par None (code nul) avec la probabilité `z_drop_prob`. La loss vaut `MSE + kl_weight · KL`, et les composantes sont exposées dans `self.last_stats` (dict).
  - `encode(points) -> Tensor` : renvoie mu.
  - `sample(joints, parents, presence, joint_mask, z=None, num_points=2048, num_steps=10, generator=None) -> tuple[list, Tensor]`, en Euler, au même format de retour que `OptimalTransportFlowMatching.sample`.
  - `configs/axl_config.yaml` contient les sections `data` (`cache_dir`, `num_points: 2048`, `animal_ratio: 0.7`, `num_workers: 6`), `model` (`hidden_dim: 256`, `num_layers: 6`, `num_heads: 8`, `z_dim: 64`, `data_scale`), `training` (lr, epochs, bf16, `save_every: 10`, `checkpoint_dir`) et `sampling`.
  - `data_scale` est fixé par une mesure. Il vaut `0.25 / std` des points de 2 000 exemples augmentés, arrondi au demi, pour retrouver l'écart-type de 1 du jeu quadrupède. La mesure va dans le commentaire de la config.
  - `scripts/train_axl.py --config` : DataLoader avec `WeightedRandomSampler(sampling_weights(groups, 0.7), num_samples=len(dataset))`, `collate_skeletons` et `persistent_workers`. Il réutilise `Trainer` (EMA, bf16, checkpoints, reprise) et affiche la loss, le KL, les échantillons par seconde et le temps par époque.

- [ ] **Step 1 : écrire les tests qui échouent**

```python
def test_dataset_pickles_without_open_memmaps(tmp_path):
    # cache factice (fixtures de l'étape 1), ds[0] lu dans le processus principal, puis pickle.loads(pickle.dumps(ds))._arrays == {}

def test_loss_from_batch_reports_fm_and_kl(tmp_path):
    # lot de 3 créatures factices de tailles différentes : loss finie, last_stats contient "fm" et "kl"

def test_creature_flow_overfits_two_creatures():
    # 2 créatures factices (quadrupède + bâton à 2 articulations), 256 points, petit modèle (hidden 64, 2 couches),
    # 300 pas Adam lr 1e-3 sur CPU : fm final < 0.5 × fm initial ; sample() renvoie (2, 256, 3) fini
```

- [ ] **Step 2 : vérifier qu'ils échouent.**
- [ ] **Step 3 : implémenter.**
- [ ] **Step 4 : vérifier** — toute la suite passe, puis un essai réel court : `scripts/train_axl.py` avec `num_epochs: 1` sur 20 fragments. On attend une loss qui baisse, et un débit consigné (échantillons par seconde, part du temps passé dans le chargement). Si le chargement prend plus de 30 % du temps, augmenter `num_workers` ou alléger l'augmentation, et consigner la décision.
- [ ] **Step 5 : commit** — `feat: creature flow matching with shape code, and Articulation-XL training script`

---

### Task 6 : évaluation des créatures

**Files:**
- Create: `scripts/evaluate_axl.py`, `tests/test_evaluate_axl.py`

**Interfaces:**
- Consumes: `CreatureFlowMatching.sample`/`encode` (Task 5), `pipeline.metrics` (MMD/COV/1-NNA existants).
- Produces:
  - `head_joint(names: list[str], joints, parents) -> int` : l'articulation dont le nom contient un mot de `HEAD_WORDS` (`data/axl/normalize.py`), sinon la feuille la plus haute.
  - `surface_gap(generated (N,3), reference (M,3)) -> float` : distance médiane de chaque point généré à son plus proche voisin dans le nuage de référence de 8192 points. C'est une approximation de la distance au maillage, notée comme telle.
  - `skeleton_sensitivity(flow, a, b) -> float` : $CD(gen(sq_A, z_B), gen(sq_B, z_B)) / CD(A, B)$. Elle mesure que le squelette compte encore quand $z$ est fixé : elle est attendue loin de 0, et une valeur sous 0.2 signale que $z$ contourne le squelette.
  - CLI `--checkpoint --config --num 200` : ce sont des créatures de test tirées avec l'équilibre 70/30.
    - Elle affiche MMD-CD, COV-CD, 1-NNA (générées/réelles) et `surface_gap` médian dans les deux modes, squelette seul et squelette plus $z$ encodé. Le second est noté « proche d'une reconstruction ».
    - Elle calcule `skeleton_sensitivity` sur 20 paires.
    - Elle écrit `docs/axl/plate_<run>.png` : 12 créatures de test fixes, une graine par tag (mammifère, oiseau, reptile/dragon, poisson/serpent, insecte, humanoïde, jouet…). Pour chacune : la réelle, la générée sans $z$, la générée avec $z$, et un gros plan de la tête (rayon 0.25 autour de `head_joint`).

- [ ] **Step 1 : écrire les tests qui échouent** :
  - `test_head_joint_prefers_named_head_then_highest_leaf`.
  - `test_surface_gap_is_zero_on_identical_clouds_and_grows_with_noise`.
  - `test_skeleton_sensitivity_on_a_toy_flow` : un flow factice qui ignore $z$ donne une sensibilité d'environ 1, un flow factice qui ignore le squelette en donne environ 0.
- [ ] **Step 2 à 4 :** vérifier qu'ils échouent, implémenter, puis faire passer toute la suite.
- [ ] **Step 5 : commit** — `feat: creature evaluation (generation metrics, skeleton sensitivity, fixed plate)`

---

### Task 7 : premier entraînement et rapport

**Files:**
- Modify: `README.md` (section « Données Articulation-XL » → résultats du premier modèle), spec (chiffres mesurés)
- Create (commitée) : `docs/axl/plate_axl_v1.png`

- [ ] **Step 1 : fixer le budget.** À partir du débit mesuré en Task 5, régler `num_epochs` pour environ 4 h de GPU. Le consigner, avec une estimation du nombre de passes sur les données.
- [ ] **Step 2 : entraîner** (`checkpoints/axl_v1`) en arrière-plan, avec reprise si le PC s'endort.
- [ ] **Step 3 : évaluer** avec `scripts/evaluate_axl.py --num 200`, sur le dernier checkpoint et sur celui de mi-parcours, pour voir si la courbe monte encore.
- [ ] **Step 4 : consigner** les deux modes, la sensibilité au squelette, les temps d'entraînement et de génération, ainsi que la planche, dans le README et la spec. Commit : `docs: first free-skeleton creature model (axl_v1) results`.
- [ ] **Step 5 : s'arrêter et montrer la planche à l'utilisateur.** C'est lui qui juge si « on lit la tête et les appendices » (critère de la spec), avant le plan 2b.
