# Étape 1 : expérience OT β, puis données de créatures Articulation-XL — plan de réalisation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** trancher la perturbation β du couplage OT sur le jeu quadrupède existant, puis produire, à partir d'Articulation-XL2.0, un cache d'entraînement de créatures (nuage dense + squelette libre + poids de peau) normalisé, avec augmentation par déformation du squelette et un rapport de contrôle.

**Architecture:** un nouveau sous-paquet `data/axl/` lit les tables Parquet du miroir `chibifire/datasource-articulation-xl2-parquet` par groupes de lignes, par requêtes HTTP Range (aucun téléchargement complet de 25 Go). Chaque groupe de lignes devient un fragment `.npz` de cache, ce qui rend la construction reprenable. Un `AXLDataset` lit ces fragments, applique la déformation par skinning et rééquilibre animaux / humanoïdes à 70/30. Le pipeline quadrupède existant n'est pas modifié, sauf l'ajout du paramètre β.

**Tech Stack:** Python 3.12 (`.venv`), PyTorch, NumPy, SciPy, pyarrow (déjà installé, à ajouter à `requirements.txt`), matplotlib (déjà utilisé par `scripts/plot_creatures.py`), urllib de la bibliothèque standard. Pas de `huggingface_hub` ni d'extension compilée.

**Spec:** `docs/superpowers/specs/2026-10-25-creatures-toutes-especes-chimeres-design.md` (étape 1 et §5 « Première expérience : la perturbation β »), éclairé par `reports/État des recherches chimères 3D.md` (parties 1 à 3).

## Global Constraints

- Python : `../3D_Creation/.venv/Scripts/python` (ou `.venv/Scripts/python` depuis la racine). Jamais le Python système. Ne jamais contourner SSL.
- Pur PyTorch / NumPy / SciPy : pas de `torch_cluster`, KeOps, PyTorch3D ni autre extension compilée.
- Tests sans GPU : `CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python -m pytest -q`. Les 48 tests existants doivent rester verts.
- Un seul job GPU à la fois : attendre que `nvidia-smi` montre moins de 500 Mo utilisés avant d'en lancer un.
- Disque : environ 30 Go libres. Le cache AXL doit tenir sous 8 Go. Ne rien supprimer dans `checkpoints/` sans l'accord de l'utilisateur.
- Source : **ensemble principal seulement** (`train/` et `test/` du miroir), jamais `diverse_pose_*` (contient des données SMALR non redistribuables).
- Convention d'orientation : **haut = +Y** (verticale de la pose d'origine), **avant = +X** (direction du regard). C'est la convention du jeu quadrupède actuel, dont la tête est en +X.
- Normalisation : centre de la boîte englobante des points à l'origine, puis une échelle unique telle que le point le plus éloigné du centre soit à distance 1.
- Équilibre : 70 % de tirages « animal » (animaux et créatures mythiques), 30 % « humanoïde » (personnages, anthropomorphes, jouets).
- Les données dérivées ne sont **pas** publiées sur Hugging Face. Le cache vit dans `data/cache/axl/`, déjà ignoré par git.
- Commits sur la branche `feat/creatures-all-species`, terminés par `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. L'utilisateur pousse lui-même.

## Review Focus

- Un modèle dont l'arbre d'os est invalide (plusieurs racines, cycle, articulation isolée, `root_index` hors bornes) : il est rejeté avec une raison consignée, sans faire tomber la construction. Test : Task 2, `test_parents_from_bones_rejects_invalid_trees`.
- Un maillage dégénéré (aire totale nulle, faces qui pointent hors des sommets, sommets NaN) : rejeté avec une raison. Test : Task 3, `test_sample_surface_rejects_degenerate_meshes`.
- Des poids de peau dont la somme par sommet est inférieure à 1, ou nulle pour certains sommets (le jeu en contient, selon ses mainteneurs) : renormalisés, et un sommet sans aucun poids est rattaché à l'os le plus proche. Test : Task 3, `test_skin_weights_are_renormalised_and_unweighted_vertices_get_nearest_bone`.
- Une construction interrompue (réseau, coupure) puis relancée : elle reprend aux fragments manquants sans dupliquer ni corrompre les fragments existants. Test : Task 6, `test_build_skips_existing_shards_and_writes_atomically`.
- Un squelette de plus de 255 articulations, ou à 2 articulations : le premier est rejeté (index stockés en `uint8`) ; le second passe toute la chaîne (échantillonnage, déformation, collate). Tests : Task 6, `test_rejects_more_than_255_joints`, et Task 7, `test_two_joint_skeleton_round_trip`.

---

### Task 1 : perturbation β du bruit apparié par OT (expérience sur le jeu quadrupède)

**Files:**
- Modify: `data/dataset.py` (constructeur de `CreatureDataset` et `__getitem__`)
- Modify: `scripts/train.py:26-27` (passer la clé de config)
- Modify: `configs/default_config.yaml` (section `data`)
- Test: `tests/test_dataset.py`
- Modify (résultats) : `README.md`, section « Couplage OT par forme »

**Interfaces:**
- Produces: `CreatureDataset(..., ot_noise_beta: float = 0.0)` ; clé de config `data.ot_noise_beta` (défaut `0.0`).

Formule (Not-So-OT, ICLR 2025) appliquée au bruit apparié choisi, **avant** l'augmentation par rotation : `noise = sqrt(1-β)·noise + sqrt(β)·ε`, avec `ε ~ N(0, I)` tiré à neuf à chaque appel. Elle est sans effet quand β = 0, et sans effet quand `ot_noise_per_sample == 0`.

- [ ] **Step 1 : écrire les tests qui échouent**

```python
def test_ot_noise_beta_zero_returns_the_matched_draw(tmp_path): ...
    # dataset avec ot_noise_per_sample=1, augment=False, ot_noise_beta=0.0
    # assert torch.equal(item["noise"], ds.ot_noise[0][0])

def test_ot_noise_beta_mixes_fresh_gaussian_noise(tmp_path): ...
    # même jeu, ot_noise_beta=0.2, augment=False, np.random.seed(0)
    # m = ds.ot_noise[0][0].flatten(); n = item["noise"].flatten()
    # assert abs(torch.corrcoef(torch.stack([m, n]))[0, 1] - 0.8 ** 0.5) < 0.02
    # assert abs(n.std() - 1.0) < 0.03
```

Utiliser de petits jeux (`num_samples_per_family=1`, `num_points=2048`, `cache_dir=tmp_path`) comme les tests existants de `tests/test_dataset.py`. Le précalcul OT d'un jeu de 5 formes prend environ 10 s ; passer `ot_workers=1`.

- [ ] **Step 2 : vérifier qu'ils échouent**

Run: `CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python -m pytest tests/test_dataset.py -k beta -v`
Expected: FAIL (`unexpected keyword argument 'ot_noise_beta'`).

- [ ] **Step 3 : implémenter le paramètre**, le passer depuis `scripts/train.py` avec `cfg["data"].get("ot_noise_beta", 0.0)`, et ajouter `ot_noise_beta: 0.0` dans `configs/default_config.yaml`, avec un commentaire d'une ligne qui renvoie à Not-So-OT.

- [ ] **Step 4 : vérifier que toute la suite passe**

Run: `CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python -m pytest -q`
Expected: 50 passed.

- [ ] **Step 5 : commit** — `feat: optional beta perturbation of OT-matched noise (Not-So-OT)`

- [ ] **Step 6 : lancer l'expérience (GPU, environ 1 h 15)**

Partir de `config_ot8_ind0_seed0.yaml` (dans le scratchpad de session ; sinon `configs/default_config.yaml` avec `num_epochs: 90`, `ot_noise_per_sample: 8`, `independent_noise_prob: 0.0`, Euler 10). Créer deux configurations, `ot_noise_beta: 0.05` et `0.2`, avec un `checkpoint_dir` propre à chacune (`checkpoints/beta005_seed0`, `checkpoints/beta02_seed0`) et `save_every: 45`. Entraîner chacune 90 époques, l'une après l'autre. Évaluer ensuite chacune, ainsi que le run de référence `checkpoints/ot8_ind0_seed1/model_epoch_090.pt` (β = 0, deuxième seed), avec `scripts/evaluate_generation.py --config <cfg> --checkpoint <ckpt>` en Euler 10. Référence connue, β = 0 seed 0 : surface médiane 0.0038, MMD 0.00092, COV 58 %, 1-NNA 53.5 % (générées 7 %).

- [ ] **Step 7 : décider et consigner**

Règle de décision : on adopte le β dont l'écart médian à la surface est le plus bas, sauf si son MMD est pire de plus de 5 % que celui de β = 0. Si l'écart à β = 0 est inférieur à 0.0002, on garde β = 0, qui est plus simple. Si un β > 0 l'emporte, le confirmer par un run à 180 époques (référence β = 0 : 0.0029/0.0030) avant de changer la valeur par défaut. Ajouter le tableau (β, surface médiane, p90, MMD, COV, 1-NNA générées/réelles) dans la section « Couplage OT par forme » du README, et mettre à jour la ligne correspondante du document de conception. La seconde dimension de l'expérience prévue par la spec (taille du superset apparié) dépend de la méthode d'OT rapide : elle est **reportée au plan de l'étape 2**, car un appariement exact à 8192 points coûte environ 64 fois plus cher.

- [ ] **Step 8 : commit** — `docs: OT beta perturbation results on the quadruped set`

---

### Task 2 : lecture distante du Parquet et arbre du squelette

**Files:**
- Create: `data/axl/__init__.py`, `data/axl/remote_parquet.py`, `data/axl/skeleton.py`
- Modify: `requirements.txt` (ajouter `pyarrow>=15.0`)
- Test: `tests/test_axl_skeleton.py`, `tests/test_axl_remote_parquet.py`

**Interfaces:**
- Produces:
  - `MIRROR_URL = "https://huggingface.co/datasets/chibifire/datasource-articulation-xl2-parquet/resolve/main/"`
  - `class RangeFile(io.RawIOBase)` : fichier distant lu par requêtes HTTP Range (suit la redirection une fois, puis réutilise l'URL finale).
  - `open_parquet(path: str) -> pyarrow.parquet.ParquetFile` : `path` est soit un chemin local existant, soit un chemin relatif au miroir (`"train/meshes.parquet"`).
  - `parents_from_bones(bones: np.ndarray, root_index: int, num_joints: int) -> np.ndarray` : tableau `int64` de taille `num_joints`, `-1` pour la racine, orienté depuis `root_index` (les paires `bones` ne sont pas supposées orientées). Lève `ValueError` avec un message court si l'arbre est invalide.
  - `tree_depth(parents: np.ndarray) -> np.ndarray` (`int64`, 0 à la racine).

Les tables du miroir partagent la même découpe : 729 groupes de 64 lignes dans le même ordre (`ordinal`) pour `models`, `skeletons`, `meshes` et `skins` en `train/`. On s'appuie dessus, et on vérifie l'égalité des `uuid` groupe par groupe (Task 6).

- [ ] **Step 1 : écrire les tests qui échouent**

```python
def test_parents_from_bones_orients_unordered_pairs_from_the_root():
    bones = np.array([[1, 0], [1, 2], [3, 2]])   # chaîne 0-1-2-3, paires dans le désordre
    assert parents_from_bones(bones, root_index=0, num_joints=4).tolist() == [-1, 0, 1, 2]
    assert parents_from_bones(bones, root_index=2, num_joints=4).tolist() == [1, 2, -1, 2]

@pytest.mark.parametrize("bones,root,n", [
    (np.array([[0, 1], [1, 2], [2, 0]]), 0, 3),  # cycle
    (np.array([[0, 1]]), 0, 3),                  # articulation 2 isolée
    (np.array([[0, 1]]), 5, 2),                  # racine hors bornes
])
def test_parents_from_bones_rejects_invalid_trees(bones, root, n):
    with pytest.raises(ValueError):
        parents_from_bones(bones, root, n)

def test_tree_depth():
    assert tree_depth(np.array([-1, 0, 1, 0])).tolist() == [0, 1, 2, 1]

def test_open_parquet_reads_a_local_file(tmp_path): ...
    # écrire une petite table pyarrow avec pq.write_table, la relire via open_parquet(str(path))
    # assert pf.metadata.num_rows == 3
```

Le test réseau de `RangeFile` est marqué `@pytest.mark.network` et ignoré par défaut : `skipif` quand la variable d'environnement `AXL_NETWORK_TESTS` est absente. Il lit `train/models.parquet` et vérifie `num_row_groups == 729`.

- [ ] **Step 2 : vérifier qu'ils échouent** — `pytest tests/test_axl_skeleton.py tests/test_axl_remote_parquet.py -v`, échec attendu à l'import.

- [ ] **Step 3 : implémenter.** `parents_from_bones` : liste d'adjacence non orientée, puis parcours en largeur depuis la racine. Il y a erreur si un sommet est atteint deux fois (cycle), si tous ne sont pas atteints, ou si `len(bones) != num_joints - 1`. `RangeFile` : `HEAD` pour la taille et l'URL finale, puis `GET` avec `Range: bytes=a-b`. Le prototype qui marche est `peek_parquet.py` dans le scratchpad de session.

- [ ] **Step 4 : vérifier** — les nouveaux tests passent, ainsi que toute la suite.

- [ ] **Step 5 : commit** — `feat(axl): remote parquet reader and skeleton tree from bone pairs`

---

### Task 3 : échantillonnage de surface avec poids de peau

**Files:**
- Create: `data/axl/sampling.py`
- Test: `tests/test_axl_sampling.py`

**Interfaces:**
- Produces:
  - `sample_surface(vertices: np.ndarray, faces: np.ndarray, num_points: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]` : `(points float32 (N,3), face_idx int64 (N,), bary float32 (N,3))`, tirage proportionnel à l'aire. Lève `ValueError` si l'aire totale est nulle, si une face référence un sommet inexistant, ou s'il y a des NaN.
  - `vertex_skin(row, col, value, n_rows, n_cols, vertices, joints, parents) -> np.ndarray` : matrice dense `float32 (V, J)` renormalisée (somme 1 par sommet). Un sommet sans poids reçoit 1 sur l'articulation enfant de l'os le plus proche (distance au segment parent → enfant ; l'os de la racine seule est la racine).
  - `point_skin(skin: np.ndarray, faces, face_idx, bary, top_k: int = 4) -> tuple[np.ndarray, np.ndarray]` : `(idx uint8 (N,k), w float32 (N,k))`. Poids interpolés de façon barycentrique, puis on garde les `k` plus forts et on renormalise.

Le format creux du miroir (`row`, `col`, `value`, `n_rows`, `n_cols`) est un COO. Vérifier sur un modèle réel si `row` indexe les sommets et `col` les articulations, ou l'inverse (`n_rows == n_vertices` ?), et l'écrire dans la docstring.

- [ ] **Step 1 : écrire les tests qui échouent**

```python
def test_sample_surface_is_area_weighted():
    # deux triangles disjoints, l'un d'aire 3x l'autre ; 20 000 points
    # assert abs((face_idx == big).mean() - 0.75) < 0.02
    # les points sont dans le plan de leur triangle (|z| < 1e-6 pour des triangles dans z=0)

def test_sample_surface_rejects_degenerate_meshes():
    # aire nulle (trois sommets colinéaires), face d'indice 99 sur 3 sommets, sommet NaN -> ValueError

def test_skin_weights_are_renormalised_and_unweighted_vertices_get_nearest_bone():
    # 3 sommets, 2 articulations (racine 0, enfant 1 en (1,0,0)) ; sommet 0 poids (0.3, 0.3), sommet 1 sans poids,
    # sommet 2 poids (0, 0.5)
    # skin[0] ≈ (0.5, 0.5) ; skin[2] ≈ (0, 1) ; skin[1] a un seul 1 sur l'articulation 1

def test_point_skin_interpolates_and_keeps_top_k():
    # un triangle dont les 3 sommets sont respectivement à 100 % sur les articulations 0, 1, 2 ; bary (0.5, 0.3, 0.2) ; top_k=2
    # idx == [0, 1], w ≈ [0.625, 0.375]
```

- [ ] **Step 2 : vérifier qu'ils échouent.**
- [ ] **Step 3 : implémenter.** NumPy vectorisé : aire par produit vectoriel, `rng.choice` avec probabilités, coordonnées barycentriques par la racine carrée (`u = 1 - sqrt(r1)`, …).
- [ ] **Step 4 : vérifier** — les nouveaux tests et toute la suite passent.
- [ ] **Step 5 : commit** — `feat(axl): area-weighted surface sampling with interpolated skin weights`

---

### Task 4 : normalisation et orientation canonique

**Files:**
- Create: `data/axl/normalize.py`
- Test: `tests/test_axl_normalize.py`

**Interfaces:**
- Produces:
  - `canonical_rotation(points: np.ndarray, joints: np.ndarray, parents: np.ndarray, names: list[str]) -> tuple[np.ndarray, str]` : rotation `float32 (3,3)` à appliquer en `x @ R`, et la méthode qui a décidé (`"name"`, `"geometry"`), consignée dans le rapport.
  - `normalize(points, joints, R) -> tuple[np.ndarray, np.ndarray, float]` : applique `R`, centre sur la boîte englobante des points, met le rayon maximal à 1, et renvoie l'échelle appliquée.

Méthode (choix laissé au constructeur par la spec, encadré ici) :
1. **Haut.** On suppose d'abord le +Y des fichiers (convention glTF). On le garde si la verticale est cohérente : les points les plus bas (5 %) appartiennent aux feuilles du squelette ou à leurs voisins, et la tête est au-dessus de la racine ou à sa hauteur. Sinon, on teste les 6 axes candidats (±X, ±Y, ±Z) avec ce même score. Les échecs connus de la littérature sont surtout des retournements de 90° ou 180°.
2. **Avant (rotation autour de +Y).** On cherche une articulation de tête par nom (`head`, `skull`, `face`, `jaw`, `beak`, `snout`, insensible à la casse, en ignorant les préfixes du type `mixamorig:`). Méthode `"name"` : l'avant est la projection horizontale du vecteur racine → tête. Sinon, méthode `"geometry"` : l'axe principal horizontal des points, orienté vers l'extrémité qui contient la feuille la plus haute.

- [ ] **Step 1 : écrire les tests qui échouent**

```python
def test_canonical_rotation_recovers_yaw_of_a_named_quadruped():
    inst = SMALQuadrupedGenerator(seed=0).generate_instance("felidae")   # tête en +X, haut en +Y
    for yaw in (0.0, 1.0, 2.5, -2.0):
        Ry = rotation_about_y(yaw)
        R, how = canonical_rotation(inst["points"] @ Ry, inst["joints"] @ Ry, PARENT_INDICES, JOINT_NAMES)
        head = (inst["joints"] @ Ry @ R)[JOINT_NAMES.index("head")]
        assert how == "name" and angle_between(head * [1, 0, 1], [1, 0, 0]) < np.radians(10)

def test_canonical_rotation_fixes_a_z_up_file():
    # même créature tournée de -90° autour de X (haut en +Z) -> après R, la tête est au-dessus des pattes (y)

def test_canonical_rotation_without_names_uses_geometry():
    # noms remplacés par "j0".."j34" -> how == "geometry", et la tête finit du côté +X (|angle| < 30°)

def test_normalize_centres_and_scales_to_unit_radius():
    # max norme == 1 (à 1e-5 près), centre de la boîte englobante == 0, joints transformés comme les points
```

Les helpers `rotation_about_y` et `angle_between` sont locaux au fichier de test. Si `JOINT_NAMES` ne contient pas exactement `"head"`, prendre le nom de tête qu'il contient.

- [ ] **Step 2 : vérifier qu'ils échouent.**
- [ ] **Step 3 : implémenter.**
- [ ] **Step 4 : vérifier** — les nouveaux tests et toute la suite passent.
- [ ] **Step 5 : commit** — `feat(axl): canonical orientation (up +Y, forward +X) and unit-radius normalisation`

---

### Task 5 : déformation par le squelette (augmentation)

**Files:**
- Create: `data/axl/augment.py`
- Test: `tests/test_axl_augment.py`

**Interfaces:**
- Consumes: `tree_depth` (Task 2) ; arrays au format de Task 3 (`idx uint8 (N,k)`, `w (N,k)`).
- Produces:
  - `@dataclass DeformParams` : `bone_scale: np.ndarray (J,)`, `girth: np.ndarray (J,)`, `rotations: np.ndarray (J,3,3)`, `presence: np.ndarray (J,)`, tous à l'identité par défaut via `DeformParams.identity(num_joints)`.
  - `random_params(parents: np.ndarray, rng, cfg: AugmentConfig) -> DeformParams`
  - `@dataclass AugmentConfig` : `bone_scale=(0.8, 1.25)`, `girth=(0.85, 1.2)`, `pose_prob=0.5`, `pose_joint_frac=0.3`, `pose_max_deg=20.0`, `subtree_prob=0.2`, `subtree_max_frac=1/3`.
  - `deform(points, skin_idx, skin_w, joints, parents, params) -> tuple[np.ndarray, np.ndarray, np.ndarray]` : `(points (N,3), joints (J,3), point_area_scale (N,))`.
  - `resample(points, point_area_scale, keep_mask, num_points, rng) -> np.ndarray` : tirage avec remise, proportionnel à `point_area_scale`, parmi les points où `keep_mask` est vrai.

Modèle de déformation (skinning linéaire, une transformation affine par os, sur les points directement) :
- L'**os j** va de `joints[parents[j]]` à `joints[j]` ; il n'existe pas pour la racine.
- On parcourt l'arbre de la racine vers les feuilles. La nouvelle position de `j` = nouvelle position du parent + `rotation_cumulée(parent) · rotations[j] · (joints[j] - joints[parent]) · bone_scale[j]`. Seule la translation descend aux enfants : l'échelle d'un os n'étire pas ses descendants (recommandation du rapport, partie 2).
- La transformation d'un point par l'os j : étirement de `bone_scale[j]` le long de l'axe de l'os, mise à l'échelle radiale `girth[j]` autour de cet axe, puis rotation et translation de l'os. Le point final est la somme pondérée par `skin_w`.
- `presence[j] < 1` sur un sous-arbre enraciné en `r` : les articulations et les points de ce sous-arbre sont ramenés vers `joints[parents[r]]` d'un facteur `presence[r]`. Avec `presence = 0`, `keep_mask` exclut les points dont le poids dominant est dans le sous-arbre.
- `point_area_scale` = Σ_k w_k · bone_scale · girth de l'os k : un point étiré représente plus de surface, et `resample` corrige ainsi la densité (rapport, partie 2).

- [ ] **Step 1 : écrire les tests qui échouent**

```python
def test_identity_params_leave_points_unchanged(): ...
    # assert np.allclose(deform(...)[0], points, atol=1e-5)

def test_bone_scale_moves_the_child_joint_and_its_rigid_points():
    # chaîne racine(0,0,0) -> j1(1,0,0) -> j2(2,0,0) ; bone_scale[1] = 1.5
    # joints'[1] == (1.5,0,0) ; joints'[2] == (2.5,0,0) (translation seulement, l'os 2 garde sa longueur)
    # un point à 100 % sur j2 en (1.5, 0.1, 0) -> (2.0, 0.1, 0)

def test_girth_scales_distance_to_the_bone_axis():
    # point à 100 % sur l'os 1, à distance 0.2 de son axe ; girth[1] = 1.5 -> distance 0.3, position axiale inchangée

def test_rotation_preserves_distances_within_a_rigid_part(): ...

def test_presence_zero_removes_the_subtree_points():
    # presence[j2] = 0 -> keep_mask faux pour les points dominés par j2, vrai ailleurs ; joints'[2] == joints'[1]

def test_resample_follows_area_scale():
    # deux points, area_scale (1, 3) -> fréquence du second ≈ 0.75 sur 20 000 tirages
```

- [ ] **Step 2 : vérifier qu'ils échouent.**
- [ ] **Step 3 : implémenter** (NumPy vectorisé sur les points, boucle Python sur les articulations ≤ 255).
- [ ] **Step 4 : vérifier** — les nouveaux tests et toute la suite passent.
- [ ] **Step 5 : commit** — `feat(axl): skeleton-driven deformation with bone length, girth, pose and subtree presence`

---

### Task 6 : sélection des créatures et construction du cache

**Files:**
- Create: `data/axl/selection.py`, `data/axl/build.py`, `scripts/build_axl_cache.py`
- Test: `tests/test_axl_selection.py`, `tests/test_axl_build.py`

**Interfaces:**
- Consumes: Tasks 2 à 4.
- Produces:
  - `ANIMAL_TAGS = {"animal", "mythical creature"}`, `HUMANOID_TAGS = {"character", "anthropomorphic", "toy"}`.
  - `group_of(tags: set[str]) -> str | None` : `"animal"` si un tag animal est présent (donc `"animal, anthropomorphic"` va côté animal), sinon `"humanoid"` si un tag humanoïde est présent, sinon `None` (écarté). Un modèle qui porte aussi un tag d'objet (`"character, weapon"`) est **gardé** : c'est une créature qui tient un objet.
  - `load_tags(meta_categories: pyarrow.Table) -> dict[str, set[str]]`.
  - `process_model(row: dict, num_points: int, rng) -> dict` : prend une ligne jointe (`uuid`, `vertices`, `faces`, `joints`, `bones`, `joint_names`, `root_index`, skin COO) et renvoie `{"uuid", "points" float16 (P,3), "skin_idx" uint8 (P,4), "skin_w" uint8 (P,4) (poids × 255 arrondis, renormalisés au chargement), "joints" float32 (J,3), "parents" int16 (J,), "names" list[str], "orient": str, "scale": float}`. Lève `ValueError(reason)` en cas de rejet.
  - `build_split(split: str, out_dir: Path, num_points: int = 8192, row_groups: Iterable[int] | None = None, open_fn=open_parquet) -> None` : un fragment `out_dir/{split}/rg{i:04d}.npz` par groupe de lignes. Les points de tous les modèles sont concaténés, avec un tableau `offsets`, et de même pour les articulations. S'y ajoutent `uuids`, `groups` et `names` (JSON), plus `out_dir/{split}/rg{i:04d}.rejects.json` (uuid → raison). L'écriture se fait dans un fichier temporaire puis par `os.replace`, et un fragment existant est sauté.
- CLI : `scripts/build_axl_cache.py --split {train,test} --out data/cache/axl --points 8192 [--row-groups 0:10] [--workers 4]` (les groupes de lignes se traitent en parallèle dans des processus).

- [ ] **Step 1 : écrire les tests qui échouent**

```python
@pytest.mark.parametrize("tags,group", [
    ({"animal"}, "animal"), ({"mythical creature", "anthropomorphic"}, "animal"),
    ({"animal", "anthropomorphic"}, "animal"), ({"character"}, "humanoid"),
    ({"anthropomorphic", "toy"}, "humanoid"), ({"character", "weapon"}, "humanoid"),
    ({"weapon"}, None), ({"anatomy"}, None), ({"vehicle"}, None), (set(), None),
])
def test_group_of(tags, group): assert group_of(tags) == group

def test_rejects_more_than_255_joints(): ...  # process_model sur une chaîne de 300 articulations -> ValueError("too many joints")

def test_build_skips_existing_shards_and_writes_atomically(tmp_path):
    # open_fn factice qui sert 2 groupes de lignes de 2 modèles synthétiques (une créature quadrupède du générateur
    # SMAL triangulée grossièrement, plus un modèle invalide) et compte les lectures
    # 1er build_split -> rg0000.npz, rg0001.npz ; le modèle invalide figure dans rejects.json
    # 2e build_split -> aucune lecture de groupe déjà présent ; un .tmp résiduel est ignoré puis remplacé
    # chaque fragment se recharge : offsets cohérents, points float16, |points| <= 1 + 1e-3
```

La fixture de maillage synthétique est un helper de test : chaque os devient un prisme triangulaire, avec des poids de peau à 100 % sur l'os. On ne construit pas de vrai maillage.

- [ ] **Step 2 : vérifier qu'ils échouent.**
- [ ] **Step 3 : implémenter.** Ordre des opérations de `process_model` :
  - arbre (Task 2) ;
  - poids des sommets (Task 3) ;
  - **orientation calculée sur les sommets** (Task 4) ;
  - échantillonnage de 8192 points ;
  - normalisation ;
  - quantification.

  Les tables lues pour un groupe de lignes sont `models` (pour `root_index`), `skeletons`, `meshes` et `skins`. Vérifier que les `uuid` sont identiques et dans le même ordre, sinon lever `RuntimeError`. Les tags viennent de `meta_categories.parquet`, lu une fois en entier (1 Mo).
- [ ] **Step 4 : vérifier** — les nouveaux tests et toute la suite passent.
- [ ] **Step 5 : essai réel sur 2 groupes de lignes** (environ 250 Mo de réseau)

Run: `.venv/Scripts/python scripts/build_axl_cache.py --split train --out data/cache/axl --row-groups 0:2`
Expected: `data/cache/axl/train/rg0000.npz` et `rg0001.npz`, un nombre de modèles retenus et rejetés affiché par groupe, une durée par groupe affichée. Si un groupe prend plus de 3 min, profiler avant de continuer.

- [ ] **Step 6 : commit** — `feat(axl): creature selection and resumable shard cache builder`

---

### Task 7 : jeu de données d'entraînement AXL

**Files:**
- Create: `data/axl/dataset.py`
- Test: `tests/test_axl_dataset.py`

**Interfaces:**
- Consumes: fragments de Task 6 ; `random_params`, `deform`, `resample` (Task 5).
- Produces:
  - `AXLDataset(cache_dir: str, split: str, num_points: int = 2048, augment: bool = True, augment_cfg: AugmentConfig | None = None, seed: int = 0)`. Les fragments sont chargés à la demande (`np.load(..., mmap_mode="r")`) et un index `(fragment, position)` est construit à l'ouverture. Attribut `groups: np.ndarray[str]`.
  - `__getitem__` → `{"points" float32 (num_points,3), "joints" float32 (J,3), "parents" int64 (J,), "presence" float32 (J,), "group" int64 (0 = animal, 1 = humanoïde), "uuid" str}`. Sans augmentation : `presence` vaut 1 et on tire `num_points` points uniformément parmi les 8192.
  - `sampling_weights(groups: np.ndarray, animal_ratio: float = 0.7) -> np.ndarray` : poids par exemple, à passer à `torch.utils.data.WeightedRandomSampler`.
  - `collate_skeletons(batch: list[dict]) -> dict` : empile les points et complète les squelettes jusqu'au `J` maximal du lot. Ajoute `joint_mask bool (B,Jmax)` ; les articulations ajoutées ont `parents = -2`, `presence = 0` et `joints = 0`.

- [ ] **Step 1 : écrire les tests qui échouent**

```python
def test_sampling_weights_give_the_requested_animal_ratio():
    groups = np.array(["animal"] * 10 + ["humanoid"] * 90)
    w = sampling_weights(groups, 0.7)
    assert abs(w[groups == "animal"].sum() / w.sum() - 0.7) < 1e-9

def test_collate_pads_skeletons_and_masks(): ...
    # deux exemples J=3 et J=5 -> joints (2,5,3), joint_mask[0] == [T,T,T,F,F], parents[0,3:] == -2

def test_two_joint_skeleton_round_trip(tmp_path):
    # fragment construit par build_split (open_fn factice) avec un modèle à 2 articulations
    # AXLDataset(augment=True) -> 10 tirages sans erreur, points de forme (2048,3) finis

def test_dataset_reads_shards_lazily_and_returns_unit_radius_points(tmp_path): ...
```

- [ ] **Step 2 : vérifier qu'ils échouent.**
- [ ] **Step 3 : implémenter.**
- [ ] **Step 4 : vérifier** — les nouveaux tests et toute la suite passent.
- [ ] **Step 5 : commit** — `feat(axl): training dataset with deformation, 70/30 sampling and skeleton padding`

---

### Task 8 : construction complète et rapport de contrôle

**Files:**
- Create: `scripts/axl_report.py`
- Create (sortie, commitée) : `docs/axl/report.md`, `docs/axl/sheet_*.png`
- Modify: `docs/superpowers/specs/2026-10-25-creatures-toutes-especes-chimeres-design.md` (chiffres mesurés, taux d'échec d'orientation)

**Interfaces:**
- Consumes: le cache complet ; `AXLDataset` (Task 7).
- Produces: `scripts/axl_report.py --cache data/cache/axl --out docs/axl [--audit 200]`.

- [ ] **Step 1 : construire le cache complet** (réseau : environ 29 Go lus en streaming, rien n'est gardé à part les fragments ; reprenable)

Run, en arrière-plan : `.venv/Scripts/python scripts/build_axl_cache.py --split train --out data/cache/axl --workers 4`, puis `--split test`.
Expected: 729 fragments `train`, 32 environ en `test`, et un cache total de moins de 8 Go (`du -sh data/cache/axl`). Si la taille projetée après 20 fragments dépasse 8 Go, s'arrêter et en parler à l'utilisateur.

- [ ] **Step 2 : écrire `scripts/axl_report.py`.** Le rapport `docs/axl/report.md` contient :
  - modèles retenus par groupe et par tag ;
  - rejets par raison ;
  - histogramme du nombre d'articulations ;
  - méthodes d'orientation (`name` / `geometry`).

  Il produit aussi trois planches PNG (matplotlib, même style que `scripts/plot_creatures.py`, vues de profil et de dessus, squelette en rouge) :
  - `sheet_animals.png` : 12 animaux tirés au hasard (seed fixe) ;
  - `sheet_humanoids.png` : 8 humanoïdes ;
  - `sheet_augment.png` : 4 créatures × 4 déformations.

  Il produit enfin `audit_*.png` : `--audit 200` créatures en pages de 25, de profil seulement, pour contrôler l'orientation à l'œil. Les pages d'audit ne sont pas commitées.

- [ ] **Step 3 : contrôler l'orientation.** Parcourir les 200 vues d'audit et compter les créatures dont le haut n'est pas +Y, ou dont la tête n'est pas du côté +X. Inscrire le taux d'échec par méthode dans `docs/axl/report.md`. Si le taux global dépasse 10 %, s'arrêter et présenter les cas typiques à l'utilisateur avant l'étape 2.

- [ ] **Step 4 : consigner** les chiffres mesurés dans la spec (nombre de modèles retenus, taille du cache, taux d'échec d'orientation) et dans le README, dans une courte section « Données Articulation-XL (en cours) ».

- [ ] **Step 5 : commit** — `docs(axl): data report, contact sheets and orientation audit`

---

## Hors de ce plan (consignés pour les plans suivants)

- La dimension « taille du superset » de l'expérience OT et la méthode d'OT rapide en pur PyTorch : plan de l'étape 2.
- L'appariement OT calculé au repos et réutilisé sous déformation : plan de l'étape 2. Il s'appuiera sur `deform` (Task 5), qui conserve l'identité des points.
- Les étiquettes de type de corps de Rig-XL (`mapping.json`), les noms lisibles et les licences de chaque objet (jointure Objaverse-XL) : plans des étapes 3 et 4 (catalogue, publication).
