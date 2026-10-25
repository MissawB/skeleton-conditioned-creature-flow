# Free datasets of rigged creatures, and how to clean and canonically orient them (state as of Oct 2026)

Scope: free datasets with mesh + skeleton + skinning for creatures (animals, mythical creatures, humanoids, toys); how Articulation-XL2.0 was built and how its users filter it; up-axis and forward canonicalisation; caption and category resources for Objaverse; licences. Research covered about 25 primary sources (arXiv HTML, HF dataset cards, GitHub READMEs). It was done under a tool-call budget, so some items are listed as gaps.

## 1. Which free datasets of rigged or articulated creatures exist, and what do they contain?

### Takeaway
Articulation-XL2.0 is the best free source that is cleanly licensed and ready to use: 48K main + ~11K diverse-pose, CC-BY-4.0, with joint names, skinning and category labels. Two others are larger or complementary. Anymate has 230K assets (Apache-2.0 on HF, noisier, 91 GB). UniRig's Rig-XL has 14.6K models, a typed body-plan taxonomy and Y-up world space, but no explicit data licence. ModelsResource-RigNet (2.7K) is small but clean and canonically oriented. SMAL-based data (SMALR, Animal3D) and DeformingThings4D are non-commercial or gated. Mixamo forbids ML training. Truebones Zoo is now paid.

### Cited Findings

**Articulation-XL2.0 (Seed3D)** (main candidate)
- Over 48K models filtered from Objaverse-XL. Deduplicated from "over 150K" rigged entries. "The quality of most data has been manually verified." Licence cc-by-4.0. Total download 39.7 GB; vertex normals added ~10 GB in the 2025-03-22 update — [HF card](https://huggingface.co/datasets/Seed3D/Articulation-XL2.0)
- Metadata fields: uuid, source, vertex_count, face_count, joint_count, bone_count, category_label, fileType, fileIdentifier. NPZ fields: vertices, faces, normals, joints, bones, root_index, uuid, pc_w_norm, joint_names, skinning_weights_{value,row,col,shape} (sparse) — [HF card](https://huggingface.co/datasets/Seed3D/Articulation-XL2.0)
- Splits: 46.7k train / 2k test (2025-03-20 release) — [MagicArticulate GitHub](https://github.com/Seed3D/MagicArticulate)
- Diverse-pose subset released 2025-08-22: 10.8k train + 500 test, which can be combined with the main set — [HF card](https://huggingface.co/datasets/Seed3D/Articulation-XL2.0). The Puppeteer repo says the total is "59.4K high-quality rigged data samples" — [Puppeteer GitHub](https://github.com/Seed3D/Puppeteer)
- Cleaning on 2025-04-18: removed 118 train + 3 test entries whose skinning rows sum to less than 1, and 2 train entries with duplicated joint names. The maintainers advise normalising skinning weights in the dataloader. The fix script is `data_utils/update_npz_rm_issue_data.py` — [HF card](https://huggingface.co/datasets/Seed3D/Articulation-XL2.0); [MagicArticulate GitHub](https://github.com/Seed3D/MagicArticulate)
- A second copy exists under `chaoyue7/Articulation-XL2.0` — [MagicArticulate GitHub](https://github.com/Seed3D/MagicArticulate)
- **Parquet mirror** `chibifire/datasource-articulation-xl2-parquet`: about 41 GB of zstd Parquet with 416,898 rows. Splits: train 46.6k, test 2k, diverse_pose_train 10.8k, diverse_pose_test 500. Tables: models / skeletons / meshes / skins / point_clouds, keyed by (split, uuid), plus `meta`, `meta_categories` (68.5k rows, one per category per model, so labels can be multiple) and `sources`. CC-BY-4.0. "Nothing is added, removed or rounded." All 46,638 train records were checked field by field against the original with 0 mismatches. The card does not say whether test or diverse-pose were checked — [HF mirror card](https://huggingface.co/datasets/chibifire/datasource-articulation-xl2-parquet)

**Articulation-XL 1.0 (MagicArticulate, CVPR 2025)** (superseded by 2.0)
- 38.8k candidates were kept down to "over 33k" models. Splits: 31.4k train / 1.6k test. Skinning subset: 28k / 1.2k (models with 55 joints or fewer) — [MagicArticulate arXiv](https://arxiv.org/html/2502.12135)

**Rig-XL + VRoid (UniRig, SIGGRAPH 2025)**
- 14,611 unique models. The pool started from a 54K Diffusion4D subset of Objaverse-XL. VRoid adds 2,061 anime humanoids from VRoidHub — [UniRig arXiv](https://arxiv.org/html/2504.12451)
- Eight classes: Mixamo, Biped, Quadruped, Bird & Flyer, Insect & Arachnid, Water Creature, Static, Other. Humans dominate. Bone counts peak at 52 (Mixamo with hands) and 28 (Mixamo without hands) — [UniRig arXiv](https://arxiv.org/html/2504.12451)
- Released at `huggingface.co/VAST-AI/UniRig/tree/main/data/rigxl`. Each model is a `raw_data.npz` with vertices, normals, faces, joints, skin (N×J dense), parents, names and matrix_local. Values are float16. A `mapping.json` links each model to its Objaverse `fileIdentifier` and type. Models are in world space with matrix_local aligned to **Y-up** (Blender convention). The repo is MIT, and no data licence is stated — [UniRig GitHub](https://github.com/VAST-AI-Research/UniRig)

**Anymate (2025)**
- 230,716 assets from Objaverse-XL (.fbx/.dae/.gltf/.glb). 438K armatured objects, of which 256K were still downloadable. About 200K remain after keeping 128 < vertices < 65,535 and 8 < bones < 64. Skeleton-similarity dedup. Assets with more than 70% of the skeleton outside the mesh were removed. Up to 3 extra keyframe poses per animated asset — [Anymate arXiv](https://arxiv.org/html/2505.06227)
- Splits: Anymate-train 225K, Anymate-test 5.6K, Anymate-small 14K (an Objaverse-1.0 subset described as "likely of higher quality"). Appendix A.1 gives 93K training samples instead, so the paper contradicts itself — [Anymate arXiv](https://arxiv.org/html/2505.06227)
- Manual inspection of 5,000 random samples found 4.6% suboptimal, mostly from skeleton issues. No category counts are given, only a word cloud covering humanoids, animals, furniture and machines — [Anymate arXiv](https://arxiv.org/html/2505.06227)
- HF `yfdeng/Anymate`: Apache-2.0, 91.3 GB, no dataset viewer — [HF card](https://huggingface.co/datasets/yfdeng/Anymate). The Apache-2.0 tag probably does not override the per-object Objaverse licences (see Q5).

**RigAnything (2025) data**
- 12,040 shapes: 2,354 RigNet + 9,686 Objaverse. Of 21,622 rigged Objaverse shapes, 811 were removed for having more than 64 joints (usually face or hair rigs) and 10,471 for low-quality rigging or skinning. 11,217 train / 823 val+test — [RigAnything arXiv PDF](https://arxiv.org/pdf/2502.09615) (via search snippet)

**ModelsResource-RigNetv1 (RigNet, 2020)** (old but still used as a benchmark)
- 2,703 models split 80/10/10 (2,163 / 270 / ~270). OBJ + rig_info TXT (joints, root, hierarchy, skinning). Remeshed to 1K–5K vertices. Hosted on Google Drive. The code is GPL-3.0; no data licence is stated — [RigNet GitHub](https://github.com/zhan-xu/RigNet)
- Average 25.0 joints (range 3–48). Models are consistently upright and front-facing. No overlap with Articulation-XL — [MagicArticulate arXiv](https://arxiv.org/html/2502.12135)

**DeformingThings4D**
- 1,972 animation sequences (1,772 animals + 200 humanoids) in 31 categories (another paper counts 39 classes). About 122,365 frames. Non-commercial research and education only. Access by Google form — [DeformingThings4D README](https://github.com/rabbityl/DeformingThings4D/blob/main/README.md); [hyper.ai listing](https://hyper.ai/en/datasets/19487)

**SMAL / SMALR / Animal3D**
- SMAL model licence: non-commercial only, **no redistribution** (including derivatives) without written permission, no training of commercial AI models — [SMAL licence](https://smal.is.tue.mpg.de/license.html)
- Animal3D: 3,379 images of 40 mammal species with 26 keypoints and SMAL pose/shape parameters. The images come from PartImageNet and COCO — [Animal3D arXiv](https://arxiv.org/abs/2308.11737v1)
- Puppeteer's diverse-pose subset contains 4.1k SMALR-generated animals (41 scans, random joint rotations within anatomical limits) — [Puppeteer arXiv](https://arxiv.org/html/2508.10898). Those samples are now in a CC-BY-4.0 dataset, which may conflict with SMAL's no-redistribution clause (see Q5).

**Quaternius (CC0)**
- "LowPoly Animated Animals" (6 animals, FBX/OBJ/Blend, CC0), "Animated LowPoly Dinosaurs" (rigged, CC0) and a rigged Fish pack — [itch.io animals](https://quaternius.itch.io/lowpoly-animated-animals); [itch.io dinosaurs](https://quaternius.itch.io/animated-lowpoly-dinosaurs); [itch.io fish](https://quaternius.itch.io/lowpoly-animated-fish)

**Mixamo** (excluded)
- Since June 2021, Adobe's terms prohibit using Mixamo content to train ML models, and redistributing raw assets is not allowed. These sources are forum and blog reproductions of the FAQ, not the official PDF — [Adobe community](https://community.adobe.com/questions-696/mixamo-license-for-machine-learning-datasets-589715); [Kinetix blog](https://kinetix.tech/blog/motion-retargeting-building-better-training-data)

**Truebones Zoo** (now paid)
- The 75+ animated animals FBX/BVH pack "is no longer FREE" and costs $100, with a "royalty free" licence. Smaller free packs appear from time to time — [Truebones Gumroad](https://truebones.gumroad.com/p/free-truebones-zoo-over-75-animated-animals-with-textures-in-fbx-format)

### Inferences
- For a free, publishable portfolio project, Articulation-XL2.0 (via the Parquet mirror) is the right backbone. UniRig's Rig-XL `mapping.json` with its type labels (Quadruped, Bird & Flyer, Insect, Water) could provide body-plan labels for Objaverse models that also appear in Articulation-XL. The join key would be the Objaverse fileIdentifier, but the overlap has not been checked.
- Anymate is about 5× larger but noisier, and at 91 GB it would be heavy on 8 GB VRAM and Kaggle/Colab disk limits. It is probably only worth a filtered subset.

### Gaps
- No per-category or joint-count histogram for Articulation-XL2.0 is published as text; the HF card has only a statistics image. It can be computed from the mirror's `meta` / `meta_categories` tables.
- VRoid Hub terms for ML use, 4DAnimals and AnimalAvatars were not verified in this pass.
- The Rig-XL download size is not stated.

## 2. Quality and composition of Articulation-XL2.0, and how papers filter and preprocess it

### Takeaway
The data is human-heavy. In 1.0's table, character (34.5%) and anthropomorphic (28.8%) dominate, against about 10% each for animal and mythical creature. Orientations vary widely, and most models are in rest pose. Every paper uses the same recipe: dedup, remove 1-bone or very large rigs (more than 100 bones, or more than 64 joints), VLM or manual quality check, normalise to a unit cube, and augment with rotation, scale and pose. Animal coverage was added artificially (SMALR) because of the humanoid skew.

### Cited Findings
- Objaverse-XL GitHub + Sketchfab subsets: 2.97M models gave 74K rigged assets, of which 48K are high quality and 26K low quality. Sketchfab: 0.89M → 64K rigged → 42K HQ. GitHub: 2.08M → 10K → 6K HQ. More than 150K duplicates removed. Bones range from 2 to 100 — [Puppeteer arXiv](https://arxiv.org/html/2508.10898)
- Filtering (1.0, kept for 2.0): dedup on skeleton + mesh similarity, remove single-joint rigs, remove rigs with more than 100 bones, and have GPT-4o rate skeleton quality from 4 rendered views. Removing the VLM filter worsened CD-J2J from 2.586 to 2.982. 2.0 adds multi-component models and removes unskinned vertices — [MagicArticulate arXiv](https://arxiv.org/html/2502.12135); [MagicArticulate GitHub](https://github.com/Seed3D/MagicArticulate); [Puppeteer arXiv](https://arxiv.org/html/2508.10898)
- Category table S7 (GPT-4o labels on renders): character 16,020; anthropomorphic 13,393; animal 4,760; mythical creature 4,734; toy 1,360; weapon 1,257; anatomy 1,227; clothing 595; misc 584; scanned 546; robot 4. These sum to about 46.5K, which does not match 1.0's 33K, so the table may describe 2.0 — [MagicArticulate arXiv](https://arxiv.org/html/2502.12135)
- 2.0 is described as humanoid-heavy, and the 4.1k SMALR animals were added to offset this. Another 7.3k diverse poses come from the Diffusion4D animation frames furthest from the rest pose — [Puppeteer arXiv](https://arxiv.org/html/2508.10898)
- "Most models are in rest pose." 2.0 has "highly varied orientations", whereas ModelsResource is upright and front-facing — [Puppeteer arXiv](https://arxiv.org/html/2508.10898)
- MagicArticulate makes no orientation assumption and trains with scale, shift and rotation augmentation. Joints are ordered spatially in z-y-x order with z vertical, which worked better than hierarchical ordering — [MagicArticulate arXiv](https://arxiv.org/html/2502.12135)
- Normalisation: mesh and skeleton go to [-0.5, 0.5]³ with the same transform. Joints are quantised to 128³. Puppeteer samples 8,192 surface points with normals, and each point takes the skin weights of its nearest vertex. Pose augmentation runs with p=0.5, each joint rotating with p=0.3 by an angle in [-60°, 60°] — [Puppeteer arXiv](https://arxiv.org/html/2508.10898)
- A loader normalisation bug that misaligned meshes and skeletons was fixed on 2025-08-07 — [MagicArticulate GitHub](https://github.com/Seed3D/MagicArticulate)
- UniRig filters: 10–256 bones, a single connected skeleton tree (which removes multi-object scenes), perceptual-hash dedup on uniformly textured renders, GPT-4o caption-based categorisation, and Mixamo pre-screening by bone names. A frequent error was bones wrongly marked "not connected", which hangs many bones directly off the root. Normalised to [-1, 1]³. Rotation augmentation up to ±30° (p=0.4) — [UniRig arXiv](https://arxiv.org/html/2504.12451). Their release also lists 31 broken training models that were removed — [UniRig GitHub](https://github.com/VAST-AI-Research/UniRig)
- Anymate's known failure modes: mesh parts that interpenetrate or sit too close together, and noisy tail joints (which they corrected) — [Anymate arXiv](https://arxiv.org/html/2505.06227)

### Inferences
- Expect three classes of problems: (a) orientation (up axis and facing) is random, (b) skin rows do not sum to 1, (c) multi-component meshes with internal geometry, which 2.0 explicitly re-admitted. For the project's "surface too thick" symptom, internal or duplicated shells in multi-component models are a likely cause. Filters worth trying: keep the largest connected components, or sample visible surfaces only (ray-cast or render-based sampling). This is my inference; none of the cited papers documents it.
- The category label is coarse (character / animal / mythical), so finer body-plan labels need Rig-XL types, Cap3D captions or joint-name heuristics.

### Gaps
- No paper reports rates of non-manifold meshes, internal geometry or scale outliers in Articulation-XL2.0.
- The supplementary criteria for MagicArticulate's GPT-4o skeleton rating were not retrieved.

## 3. Canonical orientation and normalisation (up axis and forward direction)

### Takeaway
Fully automatic orientation is still unreliable. On CanoVerse's benchmark, PCA reaches only about 23% Acc@10°. The best learned up-axis estimator reaches 89% on ShapeNet but drops to 78% on ModelNet40, with no quantitative Objaverse result. The newest large-scale effort, CanoVerse (2026), still needs a human to pick 1 of 5 candidates. For rigged data the skeleton itself is the strongest cue, since rest-pose rigs mostly follow DCC conventions (Y-up in Blender/glTF, Z-up internally in the papers). Head, spine and root joints give up and forward. Rig-based papers mostly skip canonicalisation and rely on rotation augmentation instead.

### Cited Findings
- **Symmetry-Robust 3D Orientation Estimation (ICML 2025)**. Up-axis Acc@10°: 89.2% on ShapeNet (with TTA) versus 69.5% for Upright-Net; 77.7% on ModelNet40 versus 62.3%. Failures are mostly 90° or 180° flips. Trained on ShapeNet; Objaverse results are qualitative only, because Objaverse is not canonically oriented. The flipper generalises poorly to out-of-distribution shapes. Code at github.com/cscarv/3d-orienter — [arXiv 2410.02101](https://arxiv.org/html/2410.02101v4)
- **CanoVerse (arXiv 2603.07144, 2026)**: 320K Objaverse/Objaverse-XL objects in 1,156 categories, each canonicalised to a per-category template. Up axis comes from either stable support faces with a VLM choosing the upright pose, or PCA with the sign fixed by semantic-part Chamfer distance. Front is the rotation about the vertical axis that minimises Chamfer distance (optionally with a semantic energy) to the template. A human then picks 1 of 5 candidates in under 5 s each. Of 750K objects, 51% were discarded for quality, 6% for pose errors and 43% kept — [CanoVerse arXiv](https://arxiv.org/html/2603.07144)
- CanoVerse Table 1 (Acc@10° / Acc@30°): PCA 22.98 / 34.83; VI-Net trained on Objaverse-OA 21.77 / 25.86; VI-Net trained on CanoVerse 55.31 / 64.14. Out of distribution on OmniObject3D, Orient Anything scores 4.77 / 7.32. Automatic canonicalisers (IC/GEC, lower is better): CaCa 1.532 / 1.704, ConDor 0.806 / 1.158, ShapeMatcher 0.980 / 2.072, COD 0.194 / 0.208, CanoVerse 0.032 / 0.123. Release is promised "upon acceptance" at github.com/123321456-gif/Canoverse — [CanoVerse arXiv](https://arxiv.org/html/2603.07144)
- **Objaverse-OA (NeurIPS 2025)**: 14,832 orientation-aligned models in 1,008 categories — [arXiv 2506.08640](https://arxiv.org/html/2506.08640v1) (via search summary)
- Conventions in rigging works: UniRig stores data Y-up (Blender) but sorts joints with z vertical. MagicArticulate and Puppeteer sort z-y-x with z vertical and use no canonicalisation (augmentation only) — [UniRig GitHub](https://github.com/VAST-AI-Research/UniRig); [UniRig arXiv](https://arxiv.org/html/2504.12451); [MagicArticulate arXiv](https://arxiv.org/html/2502.12135)
- Common normalisation recipes: a unit cube [-0.5, 0.5]³ (MagicArticulate, Puppeteer) or [-1, 1]³ (UniRig, Anymate), always applying the same similarity transform to the joints — [Puppeteer arXiv](https://arxiv.org/html/2508.10898); [Anymate arXiv](https://arxiv.org/html/2505.06227)

### Inferences
- A skeleton-based heuristic is the practical choice. It is my proposal, not a published method with a failure rate:
  1. Use the root→head (or spine chain) vector for up on bipeds.
  2. For quadrupeds, use the support plane under the foot or leaf joints for up, and the root→head direction projected onto the ground plane for forward.
  3. Fall back to a support-face test (CanoVerse style) when names are missing.
  4. Measure the failure rate by rendering a random sample of about 200 objects for manual review.
- Because the generator is conditioned on the skeleton, mis-oriented samples teach the model a coupling between skeleton and surface that does not depend on orientation. Canonicalisation might therefore matter less than consistent joint ordering. This is speculative.

### Gaps
- No reported failure rate exists for any up or forward detector on rigged Objaverse creatures specifically. Canonical Capsules was not reviewed directly.

## 4. Caption, name and category resources for Objaverse objects

### Takeaway
Cap3D (ODC-By, about 1.56M captions including Objaverse-XL) is the main free caption source, and its keys are Objaverse UIDs. Objaverse++ adds quality, multi-object and transparency tags. Articulation-XL already carries `category_label` and `fileIdentifier` for joining.

### Cited Findings
- Cap3D (`tiange/Cap3D`): captions for Objaverse, Objaverse-XL, ABO and ShapeNet. 785,150 Objaverse pairs, 1,562,177 in total. Released under ODC-By 1.0, but users must follow each object's licence, and some objects are not approved for commercial use — [Cap3D HF](https://huggingface.co/datasets/tiange/Cap3D)
- Objaverse++ (ICCV 2025 workshop): 10K objects manually annotated with aesthetic score, texture, multi-object and transparency attributes. A trained tagger labelled the rest, giving about 790K curated models — [Objaverse++ arXiv](https://arxiv.org/pdf/2504.07334); [CVF](https://openaccess.thecvf.com/content/ICCV2025W/CDEL/html/Lin_Objaverse_Curated_3D_Object_Dataset_with_Quality_Annotations_ICCVW_2025_paper.html)
- Rig-XL `mapping.json` gives a body-plan type per Objaverse fileIdentifier — [UniRig GitHub](https://github.com/VAST-AI-Research/UniRig)

### Inferences
- Join path: Articulation-XL `fileIdentifier` → Objaverse UID → Cap3D caption / Objaverse++ tags / Rig-XL type. Objaverse++'s multi-object flag could help filter the multi-component cases.

### Gaps
- 3DTopia captions, Objaverse LVIS category coverage of rigged objects, and the Objaverse++ data licence were not verified.

## 5. Licences of Objaverse-XL derived data: what is safe to publish

### Takeaway
Objaverse-XL as a whole is ODC-By. Each object keeps its own licence: Creative Commons on Sketchfab, and arbitrary or often null on GitHub. Some of these licences are NC, SA or GPL. The safe approach is to publish code, UIDs or fileIdentifiers and trained weights with attribution, and to republish raw meshes only for objects whose licence is CC-BY or CC0. Articulation-XL2.0's blanket CC-BY-4.0 label sits on top of these per-object terms. A SMAL-derived subset (non-redistributable) and possibly NC objects are bundled inside.

### Cited Findings
- Objaverse-XL: "The use of the dataset as a whole is licensed under the ODC-By v1.0 license", and individual objects have different licences. The `license` column is mostly null for GitHub rows; seen values include MIT and GPL-3.0. 10.85M rows — [Objaverse-XL HF](https://huggingface.co/datasets/allenai/objaverse-xl)
- Objaverse 1.0 (Sketchfab) licence split: CC-BY 721K, CC-BY-NC 25K, CC-BY-NC-SA 52K, CC-BY-SA 16K, CC0 3.5K — [objaverse PyPI](https://pypi.org/project/objaverse/0.0.4) (via search summary)
- Controversy: Sketchfab artists and the Sketchfab CEO raised concerns about scraping and about whether attribution met CC terms — [FlippedNormals blog](https://blog.flippednormals.com/objaverse-raises-concerns-about-ethics-of-scraping-3d-content/)
- Cap3D warns that per-object licences apply and some objects are not cleared for commercial use — [Cap3D HF](https://huggingface.co/datasets/tiange/Cap3D)
- SMAL: no redistribution, non-commercial only — [SMAL licence](https://smal.is.tue.mpg.de/license.html). Articulation-XL2.0's diverse-pose subset includes 4.1k SMALR models — [Puppeteer arXiv](https://arxiv.org/html/2508.10898)
- Articulation-XL2.0 and its mirror: CC-BY-4.0 — [HF card](https://huggingface.co/datasets/Seed3D/Articulation-XL2.0); [mirror](https://huggingface.co/datasets/chibifire/datasource-articulation-xl2-parquet)

### Inferences
- For the public HF Space and dataset, either:
  - publish only derived point clouds for objects whose source licence is CC-BY or CC0, filtered by joining `fileIdentifier` to the Objaverse-XL `license` column and attributing per object, or
  - publish only UIDs and the processing scripts.

  In both cases, drop the SMALR-derived diverse-pose samples from anything redistributed. This is a conservative reading, not legal advice.

### Gaps
- What fraction of Articulation-XL2.0 is NC, SA or null-licensed was not computed. It can be measured with a join on the Objaverse-XL metadata.
