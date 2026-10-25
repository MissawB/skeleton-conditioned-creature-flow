# Skeleton-driven deformation for data augmentation of rigged 3D creatures, and procedural creature generation (state as of Oct 2026)

Scope: (a) skinning methods, (b) shape augmentation through skeleton editing, (c) plausibility constraints, (d) procedural/parametric creature generators as extra data, (e) how generative and rigging papers used such augmentation, with measured gains. Project context: flow-matching point-cloud generator conditioned on a free-topology skeleton, trained on Articulation-XL2.0 (~40K rigs), one RTX 4060 laptop.

Research budget note: about 16 search/fetch calls. Several primary sources (Delta Mush original, Implicit Skinning full text, Infinigen creature docs, Generative Zoo full text) were only seen through abstracts, slides or search snippets. These are flagged where relevant.

---

## Q1. How do LBS and DQS behave under bone scaling and posing, what artefacts appear, and what fixes exist?

### Takeaway
LBS blends one scalar weight per bone, so it cannot represent stretching along a bone or twisting about it. It also collapses volume at joints (the candy-wrapper effect). DQS fixes the collapse for rigid motion, but it needs a separate two-phase pass for scale or shear and can flip or bulge. For **rest-pose data augmentation** (bone scaling + small rotations), plain LBS with a per-bone affine (scale) transform is the practical choice. Jacobson's "stretchable, twistable bones" (one extra scalar weight per bone) is the classic fix for along-bone stretch. Optimised Centres of Rotation (CoR) and Direct Delta Mush are higher-quality alternatives with precomputation costs.

### Cited Findings
- Jacobson and Sorkine, "Stretchable and Twistable Bones for Skeletal Shape Deformation" (SIGGRAPH Asia 2011 / TOG 30:165). It states that LBS with only one scalar weight function per bone cannot handle stretching (bones changing length) or twisting along the bone. The fix modifies the LBS formula with **one extra scalar weight function per bone** and needs no change to the rig or the existing weights, at constant-factor extra cost. — [SIGGRAPH history archive](https://history.siggraph.org/?p=197630); [project page](https://alecjacobson.com/weblog/2104.html)
- Kavan et al. 2007, "Skinning with Dual Quaternions": DQS removes LBS skin-collapse artefacts and uses the same rig and weights. Scale and shear need **two-phase skinning** (+29 vertex-shader instructions). Its stated limitation is **flipping artefacts**, because the skin always takes the shorter path round a joint. — [Kavan DQS page](https://users.cs.utah.edu/~ladislav/dq/)
- Disney's "Enhanced Dual Quaternion Skinning for Production Use" says DQS is not a simple drop-in replacement for LBS. It adds non-rigid extensions such as scale compensation, "scale non-compensating joints" and support joints (also covered by patent US9613456B2). — [Disney paper PDF](https://media.disneyanimation.com/uploads/production/publication_asset/98/asset/dualQ.pdf); [Patent](https://patents.google.com/patent/US9613456)
- Le and Hodgins 2016, "Real-time Skeletal Skinning with Optimized Centers of Rotation": precomputes a centre of rotation per vertex from rest pose and weights. At runtime it blends rotations (quaternion) around that centre, which reduces both LBS and DQS artefacts and stays real-time and backward-compatible. — [Disney Research paper PDF](https://la.disneyresearch.com/wp-content/uploads/Real-time-Skeletal-Skinning-with-Optimized-Centers-of-Rotation-Paper.pdf); [community C++ implementation](https://github.com/pmbittner/OptimisedCentresOfRotationSkinning)
- Delta Mush (Mancewicz et al. 2014) applies Laplacian smoothing ("mushing") to rigidly bound geometry to cut down on weight painting. Smoothing can shrink geometry and lose detail. Direct Delta Mush (Le and Lewis, SIGGRAPH 2019) gives a closed form that separates rotational and translational effects, controls bending and twisting independently, and allows skin sliding. — [EA SEED slides, Le & Lewis 2019](https://media.contentapi.ea.com/content/dam/ea/seed/presentations/le2019-siggraph2019-direct-delta-mush-skinning-and-variants.pdf); [SIGGRAPH archive](https://history.siggraph.org/?p=100174)

### Inferences
- **What happens under bone scaling in LBS.** If bone *b* is scaled by *s* along its axis in its local frame and the scale is put into its world transform T_b, vertices weighted 100% to *b* stretch correctly. Vertices blended between *b* and its parent get a linear mix of a stretched and an unstretched transform. The result is a smooth taper across the joint, which is usually acceptable. If **children inherit the parent's scale matrix** (a naïve hierarchy), the child bones are distorted too: non-uniform scale propagated through rotation gives shear. Best practice is to propagate only the *translation* of the child joint (shift children by the bone's length delta along the bone direction) and keep each bone's scale local and not inherited. This matches the "Maya segment scale compensate" convention that Disney's DQS paper reproduces with "scale non-compensating joints" (inference from the patent's naming, not verified in the full text).
- **DQS adds little for this project.** The augmentations are near rest pose with small rotations, so candy-wrapper collapse (which needs large twists of about 90° or more) will rarely trigger. LBS with per-bone affine matrices is sufficient and trivially vectorised: `X' = Σ_b w_b (A_b X + t_b)` is one `einsum` over (N points × B bones × 3×4). DQS in PyTorch is about 80–150 lines (quaternion↔dual-quaternion conversion, antipodality check, normalisation) and still needs the two-phase method for scale.
- **Stretchable bones (Jacobson & Sorkine)** is cheap to emulate. An "endpoint weight" per point (its projection parameter t∈[0,1] along the bone) lets a scaled bone stretch points proportionally to t instead of translating them rigidly. This is exactly the barycentric "position along bone" quantity the project can compute from the point-to-segment projection.
- **CoR and Delta Mush** need per-vertex precomputation over mesh connectivity (CoR integrates over triangles; Delta Mush needs a Laplacian). For point clouds you would need a kNN-graph Laplacian. That is feasible in PyTorch, but probably not worth it for augmentation, where the smoothness loss from LBS is small.

### Gaps
- I found no paper measuring LBS versus DQS error specifically under **along-bone length scaling** for augmentation purposes. The comparison above comes from method papers and reasoning.
- I did not retrieve the original Delta Mush paper (Mancewicz et al., DigiPro 2014) directly, only EA slides that describe it.

---

## Q2. How to change limb thickness/girth around a bone robustly, and which papers do shape variation by skeleton editing for augmentation?

### Takeaway
The cleanest published recipe is **ShapeBoost** (human, 2024). It parameterises shape as *bone lengths + mean width of each part slice*, where width is the distance of a vertex from its LBS-assigned bone. It uses length/width changes for augmentation and reports measurable error reductions. For creatures, the direct analogue is radial scaling about the bone axis, blended by skinning weights. Rigging papers (Puppeteer, UniRig, RigAnything) use only *pose* deformation through GT skinning as augmentation, not girth or length edits. Their ablations show clear gains.

### Cited Findings
- **ShapeBoost** (arXiv 2403.01345): splits the SMPL mesh into J=24 parts **by linear blend weights**, each with a central bone. Bone lengths are joint-pair distances, and a vertex's width is its **distance from its part's bone**. Each part is sliced into n pieces along the bone (default n=1) and slice width is the mean vertex width. — [ShapeBoost HTML](https://arxiv.org/html/2403.01345v1)
- ShapeBoost augmentation: applied to 67% of images. Aspect ratio a/b is sampled from U(0.4, 1.0) with probability 33% or U(1.0, 2.5) with probability 67% (to get chubbier bodies), via T = S·R. Vertex widths are rescaled by ab·l/l̄. Ablation (Table 6): full 11.4 mm PVE-T-SC / 25.1 P2P20K; **without augmentation 12.1 / 26.5**; without augmentation or decomposition 12.4 / 27.0. — [ShapeBoost HTML](https://arxiv.org/html/2403.01345v1); [abstract](https://arxiv.org/abs/2403.01345)
- BARC (CVPR 2022, IJCV 2023) found the original SMAL shape space **not expressive enough for many dog breeds** and modified it. It uses breed-similarity triplet and classification losses (a training signal, not geometric augmentation). Its ablations show significant shape-accuracy gains. — [BARC arXiv](https://arxiv.org/abs/2203.15536); [IJCV entry](https://ps.is.tuebingen.mpg.de/publications/barc-ijcv-2023)
- MHR (Ferguson et al. 2025, cited in SOMA) models **bone lengths explicitly** to handle skeletal ambiguity across body proportions. — [SOMA arXiv 2603.16858](https://arxiv.org/pdf/2603.16858)
- **Puppeteer** extended Articulation-XL to **59.4K rigged models, including an 11.4K diverse-pose subset**: 7.3K deformed meshes from animation frames (Diffusion4D ∩ rigged corpus, frames with maximal deviation from rest) + 4.1K **SMALR** models with "random rotation angles to SMALR animal joints while constraining angles within anatomically valid ranges" (ranges not given). About 10.9K are used for training and 500 are held out. — [Puppeteer arXiv HTML](https://arxiv.org/html/2508.10898v1)

### Inferences
- **Recommended girth operator (point level)**. For each point x with weights w_b: project onto each bone segment [j_parent, j_child] to get the foot point p_b(x) and parameter t_b. Define the radial offset r_b = x − p_b. Thickened position: x' = x + Σ_b w_b (g_b(t_b) − 1) r_b, where g_b is the per-bone girth factor, optionally varying along t (taper). Pitfalls:
  1. Points near joints project onto bone *endpoints* (t clamped to 0 or 1), so r_b then points radially away from the joint, not perpendicular to the bone. This gives "ballooning" at joints. Mitigation: use the weight blend (it naturally averages neighbouring bones), and/or clamp g toward 1 when t is near 0 or 1.
  2. Points with large weight on a bone they are far from (e.g. torso points weighted to a spine bone 0.5 body-widths away) get scaled a lot. Mitigation: cap |Δr|, or use the *dominant* bone per point as ShapeBoost does (hard assignment by weight).
  3. Root/torso bones with branching children: thickening the spine also pushes the limb attachment points outward. Child joints should then be moved with the surface (e.g. offset the child joint by the girth-scaled radial offset of its head joint), otherwise limbs embed into the body.
- **Bone-length scaling with children propagation**. Translate every descendant joint and every point weighted to the subtree by Δ = (s − 1)·(j_child − j_parent). Points of bone b itself are stretched by t_b·Δ (stretchable-bone style) rather than rigidly shifted. Points blended between b and its children then get continuous displacement.
- **Subtree shrinking/removal**: removing a subtree should also delete points whose *dominant* weight belongs to the subtree. Shrinking to scale → 0 collapses those points onto the attachment joint and creates a dense blob artefact, so dropping them (and optionally resampling to keep N constant) is preferable. No paper found that does this exactly for generative training (see Gaps).
- Puppeteer itself showed that **synthetic SMALR quadruped poses** were acceptable training data for a rigging model on Articulation-XL2.0. That supports using SMAL-family shapes as complementary data for this project, ideally for the quadruped class only.

### Gaps
- No paper found that does **girth/thickness augmentation on generic rigged meshes (non-human, non-SMAL)** for generative training. ShapeBoost (humans, 2D image warp) is the closest.
- No published angle ranges for Puppeteer's SMALR "anatomically valid ranges".

---

## Q3. Pose augmentation without joint limits: what did papers do, and how to keep poses plausible for unknown skeletons?

### Takeaway
Rigging papers on free-topology data use **random per-joint rotations within a bounded angle, applied stochastically**: UniRig ±15°, p=0.4; Puppeteer [−60°, 60°] per joint with p=0.3 per joint and p=0.5 per sample; Auto-Connect Gaussian σ=25°. Motion libraries are used only when the skeleton has a known template (Mixamo, SMAL). For unknown topologies, AnyTop (trained on Truebones Zoo, about 70 skeletons) can generate motion from skeleton structure alone. That is the main "learned prior" route, at significantly higher complexity.

### Cited Findings
- **Puppeteer** skeleton pose augmentation: probability 0.5 per sample. When applied, each joint has probability 0.3 of rotation with angles in **[−60°, 60°]**. Ablation: diverse-pose J2J/J2B/B2B 3.212/2.542/2.027 with augmentation versus **4.886/4.029/3.629 without**. On the Articulation-XL2.0 test set, J2J 3.033 versus 3.131. Skinning L1 on diverse-pose: 0.405 with augmentation versus 0.444 without (+9.6% error). Adding the diverse-pose subset further improves J2J from 3.212 to 2.514 and skinning L1 from 0.405 to 0.353. — [Puppeteer arXiv HTML](https://arxiv.org/html/2508.10898v1)
- **UniRig**: global rotation up to 30° per axis (p=0.4), scaling 0.8–1.0 (p=0.5). Mixamo/VRoid models are posed with Mixamo motion (p=0.6). For other categories it **randomly rotates individual bones within ±15° (p=0.4)**. — [UniRig arXiv](https://arxiv.org/pdf/2504.12451)
- **Auto-Connect** (RigFormer): bone perturbation with rotation angles ~ Gaussian(σ=25°), plus random translation and **non-uniform scaling**. — [Auto-Connect arXiv](https://arxiv.org/pdf/2506.11430)
- **RigAnything**: online augmentation that randomly deforms input point clouds with **ground-truth skeleton and skinning**. Its ablation (Sec. 4.4, Fig. 6) shows better generalisation to different poses. Data: RigNet 2,354 models + 9,686 filtered Objaverse models. — [RigAnything arXiv](https://arxiv.org/abs/2502.09615)
- **Anymate** (SIGGRAPH 2025): 230K rigged assets; uses LBS to deform vertices by bone poses. I found no confirmation that it uses LBS for training-time augmentation. — [Anymate arXiv](https://arxiv.org/abs/2505.06227)
- **AnyTop** (Gat et al. 2025): diffusion model that generates motion for arbitrary skeletons from skeletal structure alone. Topology is folded into attention, and textual joint descriptions give cross-skeleton semantic correspondence. Trained on **Truebones Zoo** (around 70 skeletal structures per a secondary review). It generalises with as few as three examples per topology and to unseen skeletons. — [AnyTop arXiv](https://arxiv.org/html/2502.17327v2)

### Inferences
- **Practical recipe for unknown skeletons**: per sample with p≈0.5, rotate a random ~30% of joints about a random axis by angle ~ U(−θ, θ). Use θ≈15–30° by default (UniRig level) and allow up to 60° (Puppeteer level) only for leaf or low-depth joints. Decay θ with the size of the affected subtree, since rotating a spine root by 60° swings half the body. A cheap plausibility proxy is to sample the rotation axis mostly perpendicular to the bone (bending) and keep twist about the bone axis small (≤15°). That avoids the LBS candy-wrapper regime (Q1) and roughly mimics hinge-like joints.
- **Note for a generative model**: unlike rigging (where the input mesh is posed and the target is the skeleton), here the skeleton is the *condition*. Pose augmentation must deform the skeleton condition and the points *together*. That is consistent by construction when re-skinning with the same weights.
- AnyTop or Truebones motion would give realistic poses, but it adds a heavy dependency (a pretrained motion diffusion model, joint-name text embeddings, retargeting onto Articulation-XL skeletons). Too complex for the laptop budget compared with bounded random rotations.

### Gaps
- No paper found that **learns or infers joint limits for arbitrary unseen skeletons** for augmentation. The plausibility proxies above are my inference.
- No measurement found of how large rotation ranges hurt *generative* quality (as opposed to rigging accuracy).

---

## Q4. Plausibility constraints: self-intersection, candy-wrapper, volume preservation

### Takeaway
Production-grade fixes (Implicit Skinning, elastic contact deformers, CoR, DQS) resolve candy-wrapper collapse, volume loss and skin-on-skin contact. They are costly to implement and mostly need mesh connectivity or implicit fields. For small-angle augmentation, the cheaper route is to **avoid the artefact regimes** (bounded twist, bounded bend, no extreme scale ratios) and to **reject samples** with a cheap self-intersection proxy.

### Cited Findings
- Implicit Skinning (Vaillant et al., SIGGRAPH 2013): approximates the mesh with per-bone implicit surfaces, recombines them each frame, and corrects LBS/DQS vertex positions. It models skin contact and volume compensation at joints (and optional bulges) without collision detection, in real time. — [Bath research portal](https://researchportal.bath.ac.uk/en/publications/implicit-skinning-real-time-skin-deformation-with-contact-modelin/); [paper PDF (SIGGRAPH disc mirror)](https://discmaster.textfiles.com/file/16678/SIGGRAPH%202013%20-%20Disc%203.iso/content/papers/125-0312.pdf)
- Follow-ups: joint-surface shapes need manual tuning, which a 2021 SMI paper automates. Large bends can produce self-intersections that break the implicit operators. — [IRIT STORM publications](https://www.irit.fr/recherches/STORM/MelladoNicolas/category/publications/publications-2021); [blog summary](https://solaire.cs.csub.edu/aestus/implicit-skinning-part-2/)
- Brunel, Bénard and Guennebaud 2021, "A time-independent deformer for elastic contacts": resolves contacts including self-collisions, then applies a volume-preserving procedural displacement. — [SIGGRAPH archive](https://history.siggraph.org/?p=121275)
- LBS candy-wrapper and collapse artefacts under rotation motivated DQS. DQS in turn has flipping at more than 180° and needs a two-phase method for scale. — [Kavan DQS page](https://users.cs.utah.edu/~ladislav/dq/)

### Inferences
- **Cheap self-intersection proxy for point clouds** (pure PyTorch): after deformation, for each point find kNN in the deformed cloud and check whether neighbours' **dominant bones are far apart in the kinematic tree** (graph distance ≥ 3) while they were not neighbours in rest pose. A high fraction (e.g. >1–2% of points) means interpenetration, so reject or resample. Cost: one `torch.cdist`/kNN on about 8–16K points, a few ms on GPU.
- **Volume check**: compare the convex-hull volume or an occupancy-grid volume per dominant-bone segment before and after. Large drops flag candy-wrapper collapse. This cannot be done exactly without a mesh, so it is approximate.
- Since the planned scale changes are local, LBS volume changes come mostly from the *intended* scaling. Volume preservation is not a target here: the goal is shape *variation*, not physically conserved mass.

### Gaps
- No paper found that applies Implicit Skinning or contact-aware skinning **to point clouds** for data augmentation.

---

## Q5. Point-cloud-level deformation versus deforming the mesh then sampling

### Takeaway
No paper was found that measures this directly. RigAnything and Puppeteer both deform at the point/mesh level using GT skinning, and both report gains. Analytically, the error of point-level LBS with barycentric weights is second order in triangle size, so negligible for dense samples. Point-level deformation does lose two things: **uniform surface density** after stretching, and the mesh needed for exact rejection tests.

### Cited Findings
- RigAnything deforms **input point clouds** directly with GT skeleton and skinning as online augmentation, and its ablation shows the benefit. — [RigAnything arXiv](https://arxiv.org/abs/2502.09615)
- Puppeteer articulates training samples using GT skeleton and skinning weights (pose augmentation), with large gains on the diverse-pose test (see Q3). — [Puppeteer arXiv HTML](https://arxiv.org/html/2508.10898v1)

### Inferences
- **Exactness**: for a point x = Σ_i β_i v_i on a triangle with interpolated weights w(x) = Σ_i β_i w_i, point-level LBS gives Σ_b w_b(x) T_b x. Deforming the mesh then sampling gives Σ_i β_i Σ_b w_{ib} T_b v_i. The two differ by cross terms Σ_b Σ_{i≠k} β_i β_k (w_{ib} − w_{kb})(T_b − ...)(v_i − v_k). This is of order (Δw across the triangle) × (‖T_b − T_c‖) × (edge length). It vanishes when the weights are constant on the triangle or when all T_b are equal, and is tiny for dense, small-triangle meshes. Large triangles that span joints (common in low-poly Articulation-XL assets) are the exception. Subdividing triangles longer than a threshold before sampling makes the two methods practically identical.
- **Density non-uniformity**: stretching a bone by s lowers surface point density there by about 1/s, and thickening lowers it by about 1/g. Generated clouds would then learn "sparse limbs". Fixes: (1) oversample in rest pose (e.g. 4× N), deform, then **farthest-point-sample or area-weighted resample** back to N. (2) Simpler: weight sampling probability by the expected local area stretch (s·g for points on bone b). Both are cheap in PyTorch (FPS on 64K→16K points is fine on a 4060, or use `torch_cluster.fps`).
- **Normals**: if normals are used, transform them with the inverse-transpose of the blended matrix. Under non-uniform girth scaling they are not simply rotated.
- **Implementation cost**: everything in Q1–Q5 except CoR/DQS/Implicit Skinning is about 200–400 lines of NumPy/PyTorch (forward kinematics over a parent array, segment projection, einsum blending, FPS resampling). It can run online in the DataLoader on CPU or batched on GPU.

### Gaps
- No quantitative study found comparing point-level versus mesh-level skinning augmentation.

---

## Q6. Procedural and parametric creature generators as complementary data

### Takeaway
**Infinigen** creatures (carnivores, herbivores, birds, beetles, fish) are procedurally generated **with animation rigs** in Blender, and could in principle give free rigged meshes. But export of rigged creatures (versus static OBJ/USD) is undocumented in what I could verify, and generation needs a Blender pipeline and significant CPU time. **SMAL/SMALR** is a cheap, proven source of rigged quadrupeds (Puppeteer used 4.1K SMALR samples). Its shape space is limited (BARC). Text-to-3D creature generators (DreamBeast, Muses, SK-Adapter) produce *unrigged* or skeleton-guided assets. They are interesting as skeleton-conditioned baselines, but not as free training data.

### Cited Findings
- **Infinigen** (Raistrick et al., CVPR 2023): automatically generates creature genomes, parts, assembly, materials and **animation rigs**. Parts are transpiled node-graphs or NURBS surfaces randomised under a lofting-inspired factorisation. Parts can be combined at random or genomes interpolated. Optional cloth simulation adds skin folding, sagging and motion. Automated fur grooming is included. — [Infinigen arXiv](https://arxiv.org/pdf/2306.09310)
- Infinigen creature classes: carnivores, herbivores, birds, beetles, fish, with fur and skin folding. — [CG Channel 2023](https://www.cgchannel.com/2023/07/infinigen-a-free-generator-for-procedural-3d-environments)
- Infinigen export to OBJ/FBX/USD exists since v1.4 (reported for indoor assets). Whether rigs and skinning weights export for creatures was **not confirmed**. — [CG Channel 2025](https://www.cgchannel.com/2025/06/open-source-tool-infinigen-indoors-generates-procedural-3d-interiors/); [Infinigen docs](https://infinigen.cs.princeton.edu/docs/latest/cli.html)
- **SMAL** (Zuffi et al. 2017): shape space learned from 41 toy scans that generalises to unseen quadrupeds. — [3D Menagerie arXiv](https://arxiv.org/pdf/1611.07700)
- **Generative Zoo** (Dec 2024): samples poses and shapes for many mammalian quadrupeds (SMAL family), renders and captions them, and conditions an image generator to build **GenZoo, 1M synthetic images**. A regressor trained *only* on it reaches state of the art on a real animal pose/shape benchmark. (A secondary summary claims a 57% S-MPJPE improvement on Animal3D, which I could not verify.) — [Generative Zoo arXiv](https://arxiv.org/abs/2412.08101); [alphaXiv summary (secondary)](https://alphaxiv.org/overview/2412.08101v1)
- **AWOL** (Zuffi, 2025 keynote): language-guided parametric 3D animal models; its synthetic data can train 3D shape/pose predictors across species. — [NeurIPS 2025 keynote abstract](https://neurips.cc/virtual/2025/129481)
- Creature generation papers: **DreamBeast** (SDS + part-affinity maps for fantastical part-composed animals). **DreamControl** (skeleton-conditioned text-to-3D that can be bound to the skeleton). **Muses** (Jan 2026, training-free: LLM-guided skeleton assembly → composed SLAT latents → texture, under 1 min/instance). **SK-Adapter** (2026, native 3D generation from a given skeleton via a lightweight adapter). — [DreamBeast](https://arxiv.org/abs/2409.08271); [DreamControl](https://arxiv.org/abs/2312.06439); [Muses](https://arxiv.org/abs/2601.03256); [SK-Adapter](https://arxiv.org/abs/2603.14152)

### Inferences
- **Usability ranking as free training data** for this project: (1) SMAL/SMALR random shapes and poses: easy (NumPy, a few hundred lines; SMAL model files need registration and licence acceptance on the MPI site), proven by Puppeteer, quadrupeds only. (2) The project's own procedural cylinder-around-bone generator: unlimited but **domain gap** (unrealistic surfaces). Given the memory notes (surface already too thick), mixing in cylinder data could reinforce the thickness problem, so use a low mixing ratio or drop it. (3) Infinigen: realistic and rigged, but heavy Blender pipeline, rig/weight export unverified, a few minutes of CPU per creature. Feasible for 1–5K creatures on Kaggle/Colab CPU over days. (4) Text-to-3D generators: output is unrigged (needs auto-rigging with UniRig/Puppeteer, so labels are noisy), needs a large GPU, and outputs are biased toward those models' priors.
- **Muses and SK-Adapter are the closest prior art** to the project's goal (skeleton → creature). They belong in a related-work section more than in the data pipeline.

### Gaps
- Could not confirm that Infinigen exports **skinning weights + bones** for creatures in a standard format (FBX with armature) and not just baked meshes. The docs need to be checked directly.
- No license and throughput figures found for Infinigen creature generation.
- "CreatureShop"-like tools: no relevant academic source found.

---

## Q7. Measured benefit of skeleton/pose/shape augmentation in generative or reconstruction models

### Takeaway
There are clear, quantified gains in **rigging** (Puppeteer: about 34% lower J2J error on diverse poses from pose augmentation; RigAnything ablation) and in **human shape estimation** (ShapeBoost: −0.7 mm / −1.4 mm from length/width augmentation). Purely synthetic SMAL data reaches SOTA in animal reconstruction (Generative Zoo). I found **no published measurement for point-cloud generative models conditioned on skeletons**. That is an open slot the project could fill with its own ablation.

### Cited Findings
- Puppeteer: removing pose augmentation raises diverse-pose J2J from 3.212 to 4.886 and skinning L1 by 9.6%. The diverse-pose subset improves J2J 3.212 → 2.514. — [Puppeteer](https://arxiv.org/html/2508.10898v1)
- ShapeBoost: augmentation ablation 11.4 → 12.1 mm (SSP-3D PVE-T-SC), 25.1 → 26.5 (HBW P2P20K). — [ShapeBoost](https://arxiv.org/html/2403.01345v1)
- RigAnything: online point-cloud deformation augmentation improves pose generalisation (qualitative/ablation Fig. 6). — [RigAnything](https://arxiv.org/abs/2502.09615)
- Generative Zoo: a regressor trained only on 1M synthetic SMAL-based images reaches SOTA on a real benchmark. — [Generative Zoo](https://arxiv.org/abs/2412.08101)
- AnyTop: generalises with as few as 3 examples per topology (relevant to low-data skeleton types). — [AnyTop](https://arxiv.org/html/2502.17327v2)

### Inferences
- Suggested evaluation for the project: train with and without each augmentation family (length / girth / pose / subtree removal) at equal compute. Measure (i) the existing family-classification accuracy (92% baseline for v3_ema per memory notes), (ii) Chamfer/EMD to held-out shapes, and (iii) a **skeleton-adherence metric**: the fraction of points whose nearest bone matches their skinning-dominant bone, plus per-bone radius error versus GT. The augmentations change the *joint distribution* (skeleton, shape). Gains should show most on held-out skeletons with unusual proportions, so build a test split stratified by bone-length ratios.
- Risk: girth augmentation could worsen the already "too thick" surfaces if the sampling of g is not centred on 1, or if density is not corrected (Q5). Sampling log g ~ N(0, 0.15²) clipped to [0.7, 1.4] is a reasonable starting assumption (not sourced).

### Gaps
- No skeleton-conditioned *generative* (diffusion/flow) paper found with an augmentation ablation.
- No paper found that uses **subtree shrinking/removal** as augmentation to teach partial presence of parts. The closest analogue is the part-composition in Infinigen/DreamBeast/Muses, which composes parts rather than removing them.
