# Encoding arbitrary-topology kinematic skeletons and skeleton-conditioned 3D shape generation (state as of Oct. 2026)

Project context: a Point-DiT with flow matching, currently using a 2-layer transformer over 35 fixed joints with `nn.Embedding(35)` and a point-to-joint cross-attention distance bias (λ_h = 2·2^h). Target: Articulation-XL2.0 skeletons with 2 to 100 bones, padded and masked, plus a per-joint presence/weight input.
Research budget: about 15 tool calls. Items marked **[2026]** are the most recent.

## 1. Skeleton and graph encoders for arbitrary topology: what works on small trees, and at what cost

### Takeaway
By 2025–2026 the field has converged on a standard recipe for arbitrary skeletons:
- one token per joint;
- input features built from rest-pose positions, often relative to the parent;
- a learned attention bias from the shortest-path distance (clamped near 5) and the relation type (parent, child, sibling and so on);
- optionally, embeddings of the joint names.

The 2026 works (UniMate, SAMoR) add a Laplacian spectral PE (sign-invariant RoPE), a global skeleton token injected through AdaLN, and joint-name dropout. For trees of at most 100–143 joints, the O(J²) cost is negligible: AnyTop has only 2.28M parameters.

### Cited findings
**AnyTop (Gat, Raab et al., Tel Aviv; arXiv 2502.17327, v2 June 2025, SIGGRAPH 2025)**
- Transformer denoiser in which each joint is a separate token per frame. Attention alternates between the skeleton axis and the time axis. — [AnyTop arXiv](https://arxiv.org/html/2502.17327v2)
- Per-joint features (13-D): root-relative position, 6D rotation, velocity, foot contact. The rest pose (FK with zero rotations, root-relative) is projected and prepended as an extra "frame 0" token. Maximum J = 143. — [AnyTop](https://arxiv.org/html/2502.17327v2)
- Topology-aware attention uses two J×J matrices:
  - R, the relation type, with 6 classes: child, parent, sibling, no-relation, self, end-effector;
  - D, the graph shortest-path distance.
  The model learns query and key embedding tables for D (size d_max+1) and R (size 6). The resulting terms are added to q·k before the softmax, so this is not a fixed scalar bias. — [AnyTop](https://arxiv.org/html/2502.17327v2)
- Clamp d_max = 5, close to the average kinematic chain length of 4.65. Ablation (coverage): d_max=1 gives 71.8, d_max=5 gives 80.5, d_max=10 gives 79.3. — [AnyTop](https://arxiv.org/html/2502.17327v2)
- Ablation without D/R: coverage drops from 80.5 to 76.8 and inter-diversity from 0.312 to 0.303. Without the rest-pose token: coverage 77.2. — [AnyTop](https://arxiv.org/html/2502.17327v2)
- Cost: 4 layers, F = 128, 2.28M parameters (vs 5.96M for the adapted MDM), 24 h on one RTX A6000. Skeletal augmentation costs O(J²) because distances must be recomputed. Data: Truebones Zoo, 70 skeletons, 1,219 motions. — [AnyTop](https://arxiv.org/html/2502.17327v2)

**UniMate [2026] (arXiv 2609.05415, 4 Sept. 2026)**
- Skeleton linearized by BFS and normalized by the topology diameter. Each joint's skeleton token fuses an embedding of its rest-pose position with that of its parent. — [UniMate](https://www.alphaxiv.org/abs/2609.05415.md)
- Per-head bias from lookup tables over geodesic distance and relation type (parent, sibling, ancestor), shared across frames. — [UniMate](https://www.alphaxiv.org/abs/2609.05415.md)
- Spec-RoPE: rotary angles come from the graph Laplacian's eigenvectors, passed through a sign-invariant network, instead of the joint index. — [UniMate](https://www.alphaxiv.org/abs/2609.05415.md)
- Global "topological conditioner": attention pooling over the skeleton tokens, then AdaLN-Zero in every block. In the ablations, removing it causes the largest drop in quality. — [UniMate](https://www.alphaxiv.org/abs/2609.05415.md)
- Data: UniML3D, 13,006 sequences, thousands of skeletons (bipeds, quadrupeds, birds, marine, insects, snakes, rigid objects), from Truebones, Mixamo and Objaverse-XL. Very long-tailed: 84.1% bipeds. Against AnyTop extended with text, FID falls from 2.711 to 0.757. — [UniMate](https://www.alphaxiv.org/abs/2609.05415.md)

**SAMoR [2026] (Zhang, Pons-Moll, Birdal; arXiv 2607.02148, July 2026, NeurIPS 2026)**
- Graph transformer over per-joint features, the kinematic graph and joint-name embeddings. Cross-attention pooling to K = 8 learned part tokens, then residual VQ. — [SAMoR arXiv](https://arxiv.org/abs/2607.02148); [NeurIPS poster](https://neurips.cc/virtual/2026/poster/149576)
- The decoder runs inverse cross-attention conditioned on the target skeleton, which allows transfer across topologies. — [SAMoR](https://arxiv.org/abs/2607.02148)
- An attention-supervision loss keeps the part queries from collapsing, and joint-name dropout reduces reliance on the labels. — [SAMoR](https://arxiv.org/abs/2607.02148)
- A 5.8× lower MPJPE than the variable-J baselines is reported only by a mirror site and is unverified. — [chatpaper mirror](https://chatpaper.com/es/paper/306725)

**TopoSlots [2026, unpublished]**
- A Hugging Face project index (2026-03-18) describes 79 skeletons and 24,448 motions. A slot-assignment module (138K parameters) maps joints to K = 24 slots with Sinkhorn OT. Work in progress, targeting NeurIPS 2026 or ICLR 2027. — [TopoSlots INDEX](https://huggingface.co/datasets/Tevior/TopoSlots-MotionData/blob/main/INDEX.md)

**Other motion encoders**
- SAME (SIGGRAPH Asia 2023, SNU / NC): a graph-convolution autoencoder that factors the skeleton out of the motion embedding. — [SIGGRAPH Asia 2023](https://asia.siggraph.org/2023/index.html%3Fpost_type=page&p=14494&id=papers_626&sess=sess209.html)
- HuMoT: a topology-agnostic transformer conditioned on neutral-pose templates. — [HuMoT arXiv 2305.18897](https://arxiv.org/pdf/2305.18897)
- PALUM [2026]: semantic body-part attention for retargeting. — [PALUM arXiv 2601.07272](https://arxiv.org/html/2601.07272v1)
- An ICML 2026 paper on "Semantic-Aware Motion Encoding for Topology-Agnostic Character Animation" exists. — [ICML 2026](https://icml.cc/virtual/2026/poster/62834)
- ACT ("Follow Your Track", arXiv 2606.25344 [2026]) builds on AnyTop. — [arXiv 2606.25344](https://arxiv.org/pdf/2606.25344)

**Rigging and skeleton generation (mesh → skeleton)**
- MagicArticulate (CVPR 2025): autoregressive generation in which each bone is represented by its two joints, quantized on a 128³ grid. No VQ-VAE. Point-cloud conditioning through a pretrained shape encoder. — [MagicArticulate arXiv 2502.12135](https://arxiv.org/pdf/2502.12135)
- Articulation-XL2.0: more than 48K Objaverse-XL models (46.7k train / 2k test). — [MagicArticulate](https://arxiv.org/pdf/2502.12135)
- UniRig (SIGGRAPH 2025): "Skeleton Tree Tokenization" with coordinates in 256 bins. The authors argue that naive DFS/BFS concatenation makes structural constraints hard to enforce and produces redundant tokens. Skinning uses Bone-Point Cross Attention. — [UniRig arXiv 2504.12451](https://arxiv.org/pdf/2504.12451); [GitHub](https://github.com/VAST-AI-Research/UniRig)
- Puppeteer (arXiv 2508.10898): joint-based tokenization with hierarchical ordering and stochastic perturbation. Skinning uses "topology-aware joint attention" based on skeletal graph distances. — [Puppeteer](https://arxiv.org/abs/2508.10898)
- Rigel3D [2026]: BFS-ordered joint tokens, each holding a coordinate on a 128³ grid and a parent index (parent < k). — [Rigel3D arXiv 2605.13129](https://arxiv.org/html/2605.13129)
- Branch-centric tokenization [2026]: arXiv 2609.06218 appeared in search results; its contents were not read. — [arXiv 2609.06218](https://arxiv.org/pdf/2609.06218)

**Graph positional encodings**
- Taxonomy: Laplacian-based (LapPE, SignNet), random-walk-based (RWSE, RWDIFF/LSPE) and others. Benchmark in [Benchmarking PEs for GNNs/GTs, arXiv 2411.12732](https://arxiv.org/pdf/2411.12732). GraphGPS framework: [arXiv 2205.12454](https://arxiv.org/pdf/2205.12454.pdf).
- RWSE has been especially effective with graph transformers on small molecular graphs. — [OpenReview](https://openreview.net/pdf/37b7c96d8dac7ae95eead1bb26a3616791714934.pdf)
- Shortest-path and resistance-distance RPEs differ in their power to distinguish graphs, and few works compare RPEs directly. — [arXiv 2402.14202](https://arxiv.org/pdf/2402.14202v4)
- Symmetrized LapPE, SVD-PE and Mag-PE fail to determine the walk profile on directed graphs. — [arXiv 2407.20912](https://arxiv.org/html/2407.20912v1)

### Inferences
- **Recommended for this project.** Replace `nn.Embedding(35)` with a token per joint built as MLP(rest position, parent offset, depth, presence weight). In the skeleton self-attention, add the AnyTop/UniMate bias b_h(D_ij clamped at 5) + b_h(R_ij). Optionally add LapPE with SignNet or random sign flips; Laplacian eigenvectors of a tree are cheap to compute (J ≤ 100). This is the most cited recipe, it is validated by ablations (+3.7 coverage points in AnyTop), and it is nearly free in compute.
- Laplacian eigenvectors are sign-ambiguous and degenerate for symmetric skeletons (left/right legs). That is why UniMate uses a sign-invariant network. For bilaterally symmetric creatures, the D/R distances avoid this problem.
- A global skeleton token pooled into AdaLN (UniMate) suits a Point-DiT well: the time-step AdaLN already exists, so the pooled skeleton summary can be added to it.
- Autoregressive tokenizations (UniRig, MagicArticulate, Puppeteer, Rigel3D) are built for *generating* skeletons, not encoding them. For a conditioning encoder, a set of tokens with relational biases is enough.

### Gaps
- No primary source found for Walk-the-Dog, MoMa, "Skeleton-aware networks" (Aberman et al. 2020) or RigNet within the call budget. Their mechanisms (skeletal pooling, GMEdgeConv) are not documented here.
- No head-to-head benchmark of LapPE vs RWSE vs SPD bias on *trees* of 2–100 nodes was found.
- UniRig's tokenizer details (special branch tokens) and Puppeteer's numbers were not read.

## 2. Bones vs joints: segment distance, bone tokens, local frames, parent offsets, scale, rest pose

### Takeaway
Recent works encode the **rest pose** with explicit parent information:
- UniMate fuses each joint's position with its parent's;
- AnyTop uses a root-relative rest-pose token;
- Rigel3D rasterizes features **along the bone segments**;
- UniRig and Rigel3D use bone-to-point attention for skinning.

Scale is normalized: to the unit cube or 128³/256-bin grid for meshes, by the topology diameter in UniMate.

### Cited findings
- AnyTop: rest-pose positions are root-relative and come from FK with zero rotations. Removing this token lowers coverage from 80.5 to 77.2. — [AnyTop](https://arxiv.org/html/2502.17327v2)
- UniMate: each skeleton token = embedding(rest-pose position) fused with embedding(parent position). Normalization by the topology diameter. — [UniMate](https://www.alphaxiv.org/abs/2609.05415.md)
- Rigel3D: per-bone features (skinning-weighted vertex features) are rasterized *along the bone segments* into a 64³ grid, then encoded by a sparse encoder. The mesh decoder cross-attends from surface tokens to skeleton tokens in every block. Skinning uses point–bone attention, so any number of bones is handled. — [Rigel3D](https://arxiv.org/html/2605.13129)
- MagicArticulate represents each bone by its two joints, with the mesh normalized to the unit cube. — [MagicArticulate](https://arxiv.org/pdf/2502.12135)
- UniRig: Bone-Point Cross Attention for skinning weights. — [UniRig](https://arxiv.org/pdf/2504.12451)
- SKDream encodes the 3D skeleton as 2D images through a coordinate color encoding (an XYZ → RGB map). A depth-augmented variant is compared in the supplement. — [SKDream CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_SKDream_Controllable_Multi-view_and_3D_Generation_with_Arbitrary_Skeletons_CVPR_2025_paper.pdf); [supplement](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Xu_SKDream_Controllable_Multi-view_CVPR_2025_supplemental.pdf)

### Inferences
- **Distance-to-segment bias.** The current bias ||x − J_k||² treats each joint as a point. For long bones (snakes, tails, wings) a point is closer to the *segment* [J_parent, J_k] than to its endpoints. Replacing it with the squared distance to the segment (closed form: clamp the projection t to [0,1]) amounts to a "bone token" and matches the bone-level representations of Rigel3D and UniRig. Keep both kinds of token (joints and bones) or only bones (J−1 tokens).
- **Presence weight.** To make a part fade, subtract a term such as −β·log(w_k) from the bias (or multiply the attention after the softmax). With w_k → 0, the token is ignored without changing the mask. This adapts the distance bias; no paper describes it, so it is our own proposal.
- **Scale.** Normalize the whole skeleton (bounding box or diameter) before the multi-scale biases, otherwise λ_h does not mean the same thing for an insect and a dragon. Also supply the normalized bone length as a token feature.
- Articulation-XL2.0 is in rest pose (T/A-pose) for the most part. Local frames and rotations are only needed for posed skeletons, so parent-relative offsets are enough.

### Gaps
- No ablation found that compares a joint-point bias to a segment-distance bias for shape generation.
- How to handle very different scales between rigs (millimeters vs meters in Objaverse) is not documented beyond normalization.

## 3. Joint names and semantics (text embeddings) across species: benefits and pitfalls

### Takeaway
Name embeddings help cross-skeleton generalization and joint correspondence. But they require heavy cleanup, or LLM standardization (UniMate), and they risk over-reliance, which SAMoR counters with **joint-name dropout**. AnyTop measures a mixed effect: −1.8 coverage points but +0.034 diversity.

### Cited findings
- AnyTop: joint names are encoded with T5, projected and added to the joint features at every frame. Preprocessing removes digits, symbols and redundant prefixes, normalizes left/right, translates non-English names and standardizes similar names. — [AnyTop](https://arxiv.org/html/2502.17327v2)
- AnyTop ablation: without names, coverage is 82.3 (vs 80.5) but inter-diversity is 0.276 (vs 0.312) and local diversity 0.218 (vs 0.252). The authors accept the cost in favor of cross-skeleton generalization. — [AnyTop](https://arxiv.org/html/2502.17327v2)
- AnyTop's latent space supports joint correspondence across skeletons. — [AnyTop](https://arxiv.org/html/2502.17327v2)
- UniMate: an index into a joint-name vocabulary standardized to anatomical labels with an LLM, embedding added to the token. — [UniMate](https://www.alphaxiv.org/abs/2609.05415.md)
- SAMoR: joint-name embeddings plus joint-name dropout to reduce dependence on the labels. — [SAMoR](https://arxiv.org/abs/2607.02148)

### Inferences
- For Objaverse/Articulation-XL rigs, names are often generic ("Bone.012", "mixamorig:…", Chinese or Japanese names). A practical pipeline:
  1. clean with regexes, as AnyTop does;
  2. map to a closed vocabulary (~50 anatomical classes plus "unknown") with an LLM, as UniMate does;
  3. use a learned embedding over that vocabulary rather than raw CLIP/T5 text;
  4. apply 20–50% dropout to the name embedding during training, as SAMoR does.
  At inference, names can then be missing (user-drawn chimeras).
- Names also help chimeras: "wing" from a bird and "wing" from a dragon share an embedding, which makes a part's identity transferable across species.

### Gaps
- No quantitative measure of the share of uninformative names in Articulation-XL2.0 was found.
- No direct comparison of CLIP vs T5 vs a closed vocabulary was found.

## 4. Skeleton-conditioned or rig-conditioned shape generation

### Takeaway
Few works generate shape **from** a skeleton.
- **SKDream** (CVPR 2025) is the main direct reference, but it goes through 2D: skeleton images in coordinate colors, then multi-view diffusion, then an LRM.
- **Stroke3D** [2026] lifts 2D strokes into a rigged 3D model.
- Most 2026 works (Rigel3D, AniGen) generate shape and rig *jointly* from an image.
- Articulated objects (NAP, CAGE) condition on a part graph through graph attention.

No work was found that generates a **point cloud** directly by 3D cross-attention to an arbitrary skeleton, which is a strong portfolio angle for this project.

### Cited findings
- **SKDream** (Xu, Yang, Yang; CVPR 2025):
  - a mesh skeletonization pipeline builds a large mesh–skeleton paired dataset;
  - skeletons are encoded as 2D images with coordinate color encoding;
  - a Skeletal Correlation Module injects global skeletal features into the generator;
  - multi-view generation, then a large reconstruction model, then UV refinement;
  - also evaluated on artist-made ModelResource skeletons.
  — [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_SKDream_Controllable_Multi-view_and_3D_Generation_with_Arbitrary_Skeletons_CVPR_2025_paper.pdf); [poster](https://cvpr.thecvf.com/virtual/2025/poster/35139); [supplement](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Xu_SKDream_Controllable_Multi-view_CVPR_2025_supplemental.pdf)
- **Stroke3D** [Feb. 2026] (Zhao, Zheng, Yang, Fan, Yang): "Lifting 2D strokes into rigged 3D model via latent diffusion models". It cites SKDream, MagicArticulate, Anymate, RigAnything, RigNet, CLAY and TRELLIS. Its internal mechanism could not be extracted from the compressed PDF. — [arXiv 2602.09713](https://arxiv.org/pdf/2602.09713); project page whalesong-zrs.github.io/Stroke3D_project_page/
- **Rigel3D** [May 2026] (Chatzis, Loizou, Kalogerakis):
  - image-conditioned; it jointly generates geometry, skeleton and skinning from structured latents (TRELLIS-style);
  - mesh decoder with cross-attention surface → skeleton tokens in every block;
  - does not condition on a user-supplied skeleton;
  - trained on Anymate (~225K rigs); beats Anymate and Puppeteer;
  - on ArticulationXL against AniGen, J2J falls from 10.667 to 7.168.
  — [Rigel3D](https://arxiv.org/html/2605.13129)
- **AniGen** [2026] (Huang et al.): jointly generates shape, skeleton and skinning as continuous S³ fields. The PDF was not read (larger than 10 MB). — [arXiv 2604.08746](https://arxiv.org/pdf/2604.08746); description via [Rigel3D](https://arxiv.org/html/2605.13129)
- Other related works cited by Rigel3D (motion or deformation of existing meshes, not shape generation): AnimaX, AnimaMimic, Make-It-Poseable, AnimateAnyMesh, ArtiLatent, Particulate. — [Rigel3D](https://arxiv.org/html/2605.13129)
- **NAP** (Lei et al., 2023): articulation trees as graphs; diffusion over complete graphs with a graph-attention denoiser; tasks such as Part2Motion and Motion2Part. — [NAP project](https://cis.upenn.edu/~leijh/projects/nap)
- **CAGE** (CVPR 2024): generates articulated objects from a category and a part connectivity graph. Attention modules model correlations between part attributes. Its authors argue that NAP often ignores the conditioning. — [CAGE arXiv 2312.09570](https://arxiv.org/abs/2312.09570v1); [CVPR paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_CAGE_Controllable_Articulation_GEneration_CVPR_2024_paper.pdf)
- **MIDGArD** (NeurIPS 2024) credits CAGE with pioneering end-to-end diffusion over articulation graphs. — [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/0318de478e18308a5f64297f618299d3-Paper-Conference.pdf)
- **UniMate** [2026] animates existing meshes (1.2 s per mesh vs 15.5 s for AnimateAnyMesh). This is not shape generation. — [UniMate](https://www.alphaxiv.org/abs/2609.05415.md)

### Inferences
- The project's positioning (direct 3D conditioning of a point-cloud flow on an arbitrary skeleton, without going through 2D) differs from SKDream (2D renders plus multi-view) and Rigel3D/AniGen (image → shape+rig). SKDream is the natural baseline to cite.
- CAGE suggests a useful idea: a graph-constrained attention mask among part tokens. Here it would be the skeleton bias (D, R) on the joint self-attention.

### Gaps
- Not checked within the budget: "Skeleton2Mesh", sketch/stick-figure-to-3D, SMAL/SMPL shape from skeleton, ArtGS. No primary source cited here.
- The SKDream PDF returned 403, so its exact numbers (dataset size, metrics) were not obtained.
- Stroke3D's mechanism (how the 3D skeleton conditions the latent diffusion) was not read.

## 5. Spatial conditioning mechanisms in point/3D transformers

### Takeaway
The patterns observed are:
1. cross-attention from surface tokens to skeleton tokens in every block (Rigel3D), which is close to the current design;
2. point→bone attention (UniRig, Rigel3D skinning) and topology-aware attention based on graph distance (Puppeteer);
3. relational biases learned per head (AnyTop, UniMate);
4. a global summary through AdaLN-Zero (UniMate);
5. 2D injection through image maps (SKDream).

The current multi-scale geometric bias (λ_h per head) is a sound variant and has no direct published counterpart for skeleton→points.

### Cited findings
- Rigel3D: the mesh decoder cross-attends from surface tokens to skeleton tokens in every block. The skeleton decoder cross-attends to the surface and skeleton latents after causal self-attention. — [Rigel3D](https://arxiv.org/html/2605.13129)
- UniRig: Bone-Point Cross Attention. — [UniRig](https://arxiv.org/pdf/2504.12451)
- Puppeteer: topology-aware joint attention based on skeletal graph distances. — [Puppeteer](https://arxiv.org/abs/2508.10898)
- UniMate: per-head relational bias, Spec-RoPE, and a global conditioner through AdaLN-Zero (largest ablation effect). — [UniMate](https://www.alphaxiv.org/abs/2609.05415.md)
- AnyTop: learned query/key embeddings for distance and relation, added to q·k. — [AnyTop](https://arxiv.org/html/2502.17327v2)
- SKDream: a Skeletal Correlation Module for global features plus 2D coordinate-color conditioning. — [SKDream](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_SKDream_Controllable_Multi-view_and_3D_Generation_with_Arbitrary_Skeletons_CVPR_2025_paper.pdf)

### Inferences
Recommended architecture for the generalization:
- (a) a skeleton encoder of 2–4 layers with a D/R bias plus a presence mask/weight;
- (b) point → bone cross-attention with a squared segment-distance bias and multi-scale λ_h. Scale-normalize the skeleton, and keep the −log w_k term for fading;
- (c) a pooled skeleton token (masked attention pooling) added to the AdaLN conditioning of the timestep;
- (d) for CFG, a learned "null skeleton" token.

ControlNet-style injection would only pay off with a pretrained unconditional base model, which this project does not have, so direct cross-attention is enough.

### Gaps
- No published ablation was found on multi-scale distance biases (λ_h per head) in point→joint cross-attention for generation.
- ControlNet/3D-ControlNet for point clouds was not covered within the call budget.
