# Weighted blending of 3D creatures across skeleton topologies: state of the art (as of Oct 2026)

Scope: methods and lessons for (1) joint-to-joint skeleton matching across topologies, (2) blending shape latents, (3) weighted composition in diffusion/flow models, (4) chimera/creature generation and how coherence is evaluated, (5) random-but-appealing mixture sampling and exploration UIs. Project context: a flow-matching Point-DiT conditioned on a kinematic skeleton plus a shape latent z; the target is a weighted child of 2–3 parents.

Verification legend: entries marked **[verified this session]** were confirmed via search/fetch on 2026-10-25. Entries marked **[from prior knowledge, URL not re-fetched]** are well-known papers whose URL/claims I did not re-check in this session; the writer should treat their detail claims with slightly more caution. Most recent works are marked **(2025)** / **(2026)**.

---

## Q1. Skeleton correspondence across different topologies (retargeting, graph matching, shape correspondence) — which are practical for 2–100-joint trees?

### Takeaway
The 2024–2026 animation literature has largely moved from explicit joint-to-joint mapping to *conditioning a generator directly on each skeleton's graph* plus *semantic joint names* (AnyTop 2025, UniMate 2026); where explicit correspondence is still needed, it is either sparse (Motion2Motion 2025) or semantic (LLM-normalised joint names). For a 2–100-joint tree, the practical recipe is a small cost matrix (semantic name/role + normalised position + tree features) solved by Hungarian or by fused Gromov–Wasserstein (which also respects tree structure), with unmatched joints allowed — all are milliseconds at this size.

### Cited Findings

**Retargeting / motion models across topologies**
- Skeleton-Aware Networks (Aberman et al., SIGGRAPH 2020) retarget between skeletons that are *homeomorphic* (same topology after collapsing chains), using skeletal pooling to a common primal skeleton; it does not handle genuinely different topologies (e.g. extra wings). **[from prior knowledge, URL not re-fetched]** — [arXiv 2005.05732](https://arxiv.org/abs/2005.05732)
- AnyTop (Gat, Raab et al., Tel Aviv Univ., arXiv 2502.17327, v2 June 2025) **(2025)**: a transformer diffusion model that generates motion for arbitrary skeletons from the skeleton alone; topology is integrated into attention; textual joint descriptions are injected into per-joint features so that the model "learns semantic correspondences between joints across diverse skeletons"; generalises with as few as three training examples per topology and to unseen skeletons; its latent space supports joint correspondence, temporal segmentation and editing. Earlier multi-skeleton methods relied on per-skeleton add-on blocks or separate model instances. **[verified]** — [arXiv HTML](https://arxiv.org/html/2502.17327v2)
- UniMate (Mou, Lei, Dou, Cai, Song, Finkelstein, Rusinkiewicz; Princeton/Berkeley/MIT/NTU; arXiv 2609.05415, 4 Sep 2026) **(2026, most recent)**: flow-matching DiT ("TADiT") conditioned on a rest-pose skeleton + text. Each skeleton is canonicalised by BFS joint ordering and scaled by its topology diameter. Topology injected three ways: (a) graph-aware attention bias from pairwise joint relations/geodesic (hop) distances, (b) "Spec-RoPE" — rotary embeddings whose angles come from graph-Laplacian eigenvectors instead of joint indices, (c) a global topology token attention-pooled from skeleton tokens and injected via AdaLN-Zero. Joint names are mapped by an LLM to a fixed anatomical vocabulary and embedded per joint; an LLM also picks a symmetric joint pair to define the facing direction. It does **not** compute explicit joint-to-joint correspondences. Dataset UniML3D: 13,006 sequences (~2.14M frames) from Truebones, Mixamo, Objaverse-XL. vs AnyTop+text: FID 2.711→0.757. Ablation: removing graph bias / Spec-RoPE / global conditioner raises FID to 0.773 / 0.798 / 0.825. Limitations: weaker on rare topologies. **[verified]** — [alphaXiv 2609.05415](https://www.alphaxiv.org/abs/2609.05415.md)
- Motion2Motion (arXiv 2508.13139, Aug 2025) **(2025)**: cross-topology motion transfer that argues "inherent topological inconsistency" between source and target skeletons blocks one-to-one bone mapping, and works from a *sparse* set of correspondences instead. **[verified, abstract-level]** — [arXiv 2508.13139](https://arxiv.org/html/2508.13139v1)
- PALUM (arXiv 2601.07272, Jan 2026) **(2026)**: part-based attention for unified retargeting, learning skeleton-agnostic motion features conditioned on the target structure; argues global attention struggles with varying joint counts (hence grouping joints into body parts). **[verified, abstract-level]** — [arXiv 2601.07272](https://arxiv.org/html/2601.07272v1)
- "Semantic-Aware Motion Encoding for Topology-Agnostic Character Animation" is an ICML 2026 poster **(2026)**; only the title was surfaced. **[verified title only]** — [ICML 2026](https://icml.cc/virtual/2026/poster/62834)
- SAME: Skeleton-Agnostic Motion Embedding (Lee, Kang, Park, Lee, Won; SIGGRAPH Asia 2023): graph-convolution autoencoder learning an embedding that disentangles skeleton information from motion semantics; used for retargeting, reconstruction, interactive control. **[verified]** — [SIGGRAPH Asia 2023 program](https://asia.siggraph.org/2023/index.html%3Fpost_type=page&p=14494&id=papers_626&sess=sess167.html)
- HuMoT (2023): transformer autoencoder conditioned on skeleton templates; supports topologies unseen in training. **[verified]** — [arXiv 2305.18897](https://arxiv.org/pdf/2305.18897)
- Masked-pose-modelling retargeting (Martinelli et al., ECCV'24 workshop, LNCS 2025) retargets to different skeletons at test time incl. human↔quadruped. **[verified, snippet]** — [IRIS UniTN](https://iris.unitn.it/handle/11572/465957)
- WalkTheDog (Li, Starke, Ye, Sorkine-Hornung, SIGGRAPH 2024): unsupervised shared *phase manifold* (several closed curves, each with a latent amplitude, VQ periodic autoencoder) for e.g. human and dog; semantically similar motions land on the same curve and are time-aligned by phase — correspondence in *motion* space, not joint space. **[verified]** — [arXiv 2407.18946](https://arxiv.org/pdf/2407.18946)
- MoMa: no source found in this session (see Gaps).

**Graph matching**
- Gromov–Wasserstein learning for graph matching (Xu, Luo, Zha, Carin, ICML 2019): the GW discrepancy compares two graphs via intra-graph distance matrices and yields a soft node correspondence (transport plan); jointly learns node embeddings. **[verified]** — [arXiv 1901.06003](https://arxiv.org/pdf/1901.06003)
- Fused Gromov–Wasserstein (Vayer et al.) combines a feature cost (like Hungarian on features) with a structure cost (like GW on intra-graph distances) under one transport plan, a natural fit for "position + tree role + hop distance" joint matching; implemented in the POT library (`ot.gromov.fused_gromov_wasserstein`), including unbalanced/partial variants useful when some joints should remain unmatched. **[from prior knowledge, URL not re-fetched]** — [arXiv 1805.09114](https://arxiv.org/abs/1805.09114); [POT docs](https://pythonot.github.io/)

**Shape correspondence between different animals**
- Zero-Shot 3D Shape Correspondence (Abdelreheem, Eldesokey, Ovsjanikov, Wonka, SIGGRAPH Asia 2023): for strongly non-isometric, inter-class pairs (human vs cow), classify the shape from renders with a VLM, ask an LLM for semantic regions and a region-to-region mapping, then refine with functional maps into dense maps; introduces the SNIS benchmark. **[verified]** — [arXiv 2306.03253](https://arxiv.org/abs/2306.03253v1); [PDF](https://www.lix.polytechnique.fr/~maks/papers/ZeroShotCorr_SIGGAsia2023.pdf)
- Diff3F (Dutt, Muralikrishnan, Mitra, CVPR 2024): renders untextured meshes/point clouds to depth/normal, uses them to condition image diffusion, lifts and aggregates 2D foundation features back onto 3D points; no fine-tuning; correspondence on SHREC'19/'20, FAUST, TOSCA including non-isometric families. **[verified]** — [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Dutt_Diffusion_3D_Features_Diff3F_Decorating_Untextured_Shapes_with_Distilled_Semantic_CVPR_2024_paper.html)
- SMAL (Zuffi et al., "3D Menagerie", CVPR 2017): a single parametric quadruped model (shared skeleton/mesh) fitted across cats, dogs, horses, cows, hippos — gives correspondence *for free* within quadrupeds, but not to birds/dragons/humanoids. **[from prior knowledge, URL not re-fetched]** — [arXiv 1611.07700](https://arxiv.org/abs/1611.07700)

### Inferences
- For the project, an explicit correspondence is still required (to interpolate joint positions), unlike UniMate/AnyTop which only need conditioning. Best-supported design: per-joint descriptor = [LLM- or rule-normalised anatomical label (AnyTop/UniMate style), position normalised by topology diameter (UniMate), depth/hop distance from root, chain role (limb/tail/neck/wing), Laplacian eigenvector coords (Spec-RoPE idea)] → cost matrix → Hungarian with a "dummy/unmatched" column, or partial fused-GW to also penalise breaking parent-child structure. At ≤100 joints both run in milliseconds.
- Matching at *chain/part level first* (PALUM's part grouping; Zero-Shot Correspondence's region mapping) then joint-level within matched chains (resample chains to equal length by arc length) is more robust than flat joint matching, and avoids crossing assignments (a lion's front leg matched to a dragon's wing).
- Since the project's own generator is skeleton-conditioned like UniMate, adopting UniMate's topology encodings (hop-distance attention bias, Laplacian RoPE, diameter normalisation) would likely make the generator more robust to the novel, interpolated skeletons the blending produces.

### Gaps
- No source found for "MoMa" in this session; could not verify what it is.
- No paper found that performs *weighted interpolation of two skeletons with different topologies* (as opposed to retargeting motion between them); this appears to be a gap the project would fill.
- No benchmark comparing Hungarian vs fused-GW vs tree edit distance specifically on animal skeletons was found.

---

## Q2. Shape interpolation/morphing between dissimilar shapes; keeping linear latent combinations in-distribution

### Takeaway
Latent interpolation is a standard demo in 3D generative models (IM-Net, DeepSDF, LION, 3DShape2VecSet, Shap-E), but between dissimilar categories it tends to pass through off-manifold, low-density regions; the robust fixes are norm-aware interpolation of Gaussian-like latents, interpolating in *noise/inverted* space rather than intermediate latents, and making the conditioning do the structural work (skeleton) so z only carries style. OT/Wasserstein barycenters give geometrically sensible weighted morphs of measures (point clouds) but no semantics.

### Cited Findings
- LION (NVIDIA, 2022) reports shape autoencoding and latent shape interpolation among its capabilities. **[from prior knowledge, URL not re-fetched]** — [arXiv 2210.06978](https://arxiv.org/abs/2210.06978)
- 3DShape2VecSet (SIGGRAPH 2023): set-of-latent-vectors representation (cross-/self-attention, RBF-inspired) designed for latent diffusion; authors found a compressed latent space much easier for 3D diffusion. **[verified]** — [arXiv 2301.11445](https://arxiv.org/pdf/2301.11445)
- Part-based latent spaces often interpolate in discrete steps (structural switching) rather than smooth deformation (search-result summary, generic claim). Part-latent generators like PartCrafter (NeurIPS 2025) and CoPart (ICCV 2025, "contextual part latents" with mutual guidance for coherent joint part denoising) represent objects as multiple part latents denoised jointly. **[verified, abstract-level]** — [CoPart ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Dong_From_One_to_More_Contextual_Part_Latents_for_3D_Generation_ICCV_2025_paper.pdf); [PartCrafter NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/file/32cc61322f1e2f56f989d29ccc7cfbb7-Paper-Conference.pdf)
- LAMP (arXiv 2510.22491, 2025) **(2025)**: linear affine *weight-space* models for parameter-controlled 3D shape generation and extrapolation — a linear-mixing approach in weight space rather than latent space. **[verified title]** — [arXiv 2510.22491](https://arxiv.org/pdf/2510.22491)
- Convolutional Wasserstein Distances (Solomon, de Goes, Peyré, Cuturi et al., SIGGRAPH 2015): entropic OT via heat-kernel convolutions; weighted Wasserstein barycenters give a cow→duck→torus morph on a 100³ grid. A later comparison reports it fails for small entropy at some weight pairs (blurring trade-off). **[verified]** — [paper](https://graphics.pixar.com/library/OptimalTransportation/paper.pdf); [comparison arXiv 2312.00327](https://arxiv.org/pdf/2312.00327)
- Slerp vs lerp for diffusion latents: linear interpolation of Gaussian noise shrinks norms into low-probability regions, so slerp is used (arXiv 2508.01698); but "Addressing degeneracies in latent interpolation for diffusion models" (arXiv 2505.07481, 2025) shows the common normalised-angle slerp can push norms outside the input range and proposes setting the interpolant's norm from the linearly interpolated input norms. **[verified, snippet]** — [arXiv 2508.01698](https://arxiv.org/pdf/2508.01698); [arXiv 2505.07481](https://arxiv.org/html/2505.07481v1)
- "There and Back Again" (arXiv 2410.23530): interpolating in *noise* space preserves image quality along the path, while interpolating DDIM-inverted intermediate latents degrades. **[verified, snippet]** — [arXiv 2410.23530](https://arxiv.org/pdf/2410.23530)
- "Going beyond Compositions, DDPMs Can Produce Zero-Shot Interpolations" (arXiv 2405.19201): diffusion models can generate plausible samples *between* training attribute values (interpolation on the condition) beyond pure composition. **[verified title/snippet]** — [arXiv 2405.19201](https://arxiv.org/pdf/2405.19201)
- Smooth Diffusion: small latent perturbations can produce visible jumps in diffusion outputs; proposes training for latent smoothness to improve interpolation/inversion/editing (seen only via aggregator). **[weakly verified]** — [aggregator](https://feddit.it/post/3895555)

### Inferences
- For the project, z = Σ wᵢ zᵢ is safe only if z's encoder is regularised (KL/VAE or a learned prior). Recommended mitigations, in order of cost: (a) rescale the blended z to the expected norm (2505.07481 rule: ‖z‖ = Σ wᵢ‖zᵢ‖); (b) project the blend back on-manifold, e.g. by a few steps of a small prior/flow on z, or by nearest-neighbour regularisation; (c) blend in the generator's *initial noise* with the same coupling rather than in intermediate states (2410.23530). Since the project uses per-shape OT coupling, sharing (slerp-blending) the starting noise across parents should also reduce random detail differences.
- Wasserstein barycenters of the parents' point clouds could serve as a *target/guide* or as an evaluation reference ("is the child near the W-barycenter?"), but used directly they produce ghosting/averaged geometry (e.g. two half-tails) rather than a fused anatomy.

### Gaps
- Could not verify specific off-manifold failure studies for 3D latent interpolation between categories (e.g. holes in IM-Net/DeepSDF blends); no direct source fetched for DeepSDF/IM-Net/Shap-E interpolation results.
- No 2025–2026 paper found on in-distribution guarantees for latent *mixtures of 3+ shapes*.

---

## Q3. Compositional generation and weighted mixing in diffusion/flow models

### Takeaway
Weighted sums of scores/velocities (Composable Diffusion, multi-condition CFG) are the standard "product of experts" approximation, but naive summation is not the score of the true product and can fail; MCMC/energy-based correction (Du et al. 2023) fixes sampling. In T2I, the best concept-fusion results use *timestep-scheduled* or *attention-level* mixing (MagicMix: layout from A early, semantics of B later; AID: attention interpolation with beta schedule; ATIH: balancing cross-/self-attention), and a 2025 study finds no single method wins across pairs.

### Cited Findings
- Composable Diffusion (Liu, Li, Du, Torralba, Tenenbaum, ECCV 2022): AND/NOT of concepts by summing weighted conditional score differences (ε_uncond + Σ wᵢ(ε_i − ε_uncond)) — equivalent to multi-condition CFG. **[from prior knowledge, URL not re-fetched]** — [arXiv 2206.01714](https://arxiv.org/abs/2206.01714)
- Reduce, Reuse, Recycle (Du et al., ICML 2023): attributes composition failures largely to the *sampler*, not the model; proposes energy-based parameterisation and MCMC (incl. Metropolis-corrected) samplers that make product/mixture composition work. **[verified]** — [arXiv 2302.11552](https://arxiv.org/pdf/2302.11552); [PMLR](https://proceedings.mlr.press/v202/du23a.html)
- MagicMix (Liew et al., ByteDance, 2022): semantic mixing (e.g. "corgi-like coffee machine") with no masks or retraining by exploiting that layout forms in early denoising steps and semantics later: generate coarse layout with concept A, then switch the prompt to concept B. **[verified]** — [project](https://magicmix.github.io/); [arXiv 2210.16056](https://arxiv.org/pdf/2210.16056)
- AID: Attention Interpolation of Text-to-Image Diffusion (He, Wang, Liu, Yao, NeurIPS 2024): linear interpolation of text embeddings gives inconsistent, low-fidelity intermediates; AID interpolates inside attention (inner/outer) and uses a beta-distribution schedule for the coefficients for smoothness; PAID adds prompt-guided path control. **[verified]** — [NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/b12a1d1014e952e676f5d6931d03241a-Abstract-Conference.html)
- Blending Concepts with T2I Diffusion Models (Olearo et al., arXiv 2506.23630, 2025; journal version 2026) **(2025/2026)**: compares four zero-shot blending methods acting on different pipeline stages (e.g. prompt-embedding interpolation, prompt switching, etc.); 100-participant user study: no method dominates; prompt order, conceptual distance and random seed all matter; outcomes depend heavily on the concept pair (animal–animal). Code public. **[verified, abstract-level]** — [arXiv 2506.23630](https://arxiv.org/abs/2506.23630); [Springer 2026](https://link.springer.com/article/10.1007/s44163-026-01134-1); [GitHub](https://github.com/LorenzoOlearo/blending-diffusion-models)
- FreeBlend (arXiv 2502.05606, 2025) **(2025)**: feedback-driven latent interpolation for concept blending using image embeddings. **[verified, snippet]** — [arXiv 2502.05606](https://arxiv.org/abs/2502.05606)
- ATIH — Novel Object Synthesis via Adaptive Text-Image Harmony (Xiong et al., NeurIPS 2024): fuses an object text with an object image; tunes a cross-attention scale and a self-attention injection step via Golden Section Search on a DINO+CLIP similarity balance so neither input dominates; preferred ~80% in fusion user study (95 participants). **[verified]** — [project](https://xzr52.github.io/ATIH/); [arXiv 2410.20823](https://arxiv.org/pdf/2410.20823)
- Self-Cross Diffusion Guidance (arXiv 2411.18936) documents the *unwanted* version of blending: "a bear and an elephant" prompts often merge into one hybrid; it separates attention maps — useful as a diagnostic of how attention overlap causes fusion. **[verified, snippet]** — [alphaXiv 2411.18936](https://www.alphaxiv.org/abs/2411.18936)
- TP-Blend (arXiv 2601.08011, Jan 2026) **(2026)**: textual-prompt attention pairing for object–style blending. **[verified title]** — [arXiv 2601.08011](https://arxiv.org/pdf/2601.08011)
- "Redefining <Creative> in Dictionary" (arXiv 2410.24160): learns a "creative" token for creative generation in T2I. **[verified title]** — [arXiv 2410.24160](https://arxiv.org/pdf/2410.24160)

### Inferences
- Direct transfer to the project's flow model: with conditions c_i = (skeleton_i, z_i), the velocity mixture v = Σ wᵢ v(x,t|c_i) is the flow analogue of Composable Diffusion; however, since the parents have *different skeletons*, a velocity mix will try to grow both bodies in the same space (ghosting). Better: condition on the single blended skeleton (structure) and mix only z (or mix velocities across z_i with the *same* blended skeleton), i.e. weighted multi-z CFG. 
- MagicMix's lesson maps to a time schedule: early steps (coarse mass distribution) governed by the blended skeleton and dominant parent's z, late steps (surface details: mane, scales, face) with weights shifted toward the minority parent — gives "fused" detail rather than averaged blobs.
- AID's beta schedule suggests mapping the user's linear slider through a non-linear (beta CDF) curve, because perceived trait share is non-linear in latent weight.

### Gaps
- No 3D/point-cloud paper found that evaluates weighted velocity/score mixing in a flow model for shape blending.
- Could not locate "Black Sheep in the Herd", "TP2O", "Combinational creativity" or "CreativeSynth" via search in this session; their exact claims are unverified.

---

## Q4. Creature/chimera generation: methods, coherence vs patchwork, evaluation

### Takeaway
Most hybrid-creature work is 2D: part-token approaches (DreamCreature, Chimera 2025) explicitly *compose parts* (risking patchwork, mitigated by joint generation in a shared diffusion prior), while concept-blending approaches (MagicMix, ATIH, Olearo 2025) produce fused wholes but with low controllability. 3D work focuses on anatomically controllable animals (YouDream) and part-aware generation (PartGen, CoPart, PartCrafter); no 2025–2026 paper was found doing weighted 3D hybrid creatures from skeletons. Evaluation is ad hoc: user studies, CLIP/DINO similarity balance, and part-alignment metrics (PartEval).

### Cited Findings
- DreamCreature (arXiv 2311.15477): learns sub-concept (part) tokens from unlabelled images of a category (e.g. birds, dogs) and composes them to create new hybrid creatures (e.g. one species' head with another's body). **[verified PDF link; detail from prior knowledge]** — [arXiv 2311.15477](https://arxiv.org/pdf/2311.15477)
- Chimera: Compositional Image Generation using Part-based Concepting (Singh, Chen, Chatterjee, Raj, Hays, Yang, Baral; arXiv 2510.18083, Oct 2025) **(2025)**: assembles new objects from specified parts of several source images guided by text, no masks; taxonomy of 464 (part, subject) "semantic atoms" → 37k prompts/images; trains a custom diffusion prior with *part-conditional guidance* enforcing semantic identity and spatial layout; introduces **PartEval** (fidelity and compositional accuracy); +14% part alignment/compositional accuracy and +21% visual quality over baselines. **[verified]** — [arXiv 2510.18083](https://arxiv.org/abs/2510.18083)
- YouDream (NeurIPS 2024): text-to-3D animals guided by a 3D pose prior (skeleton) rendered into 2D control views — anatomically consistent, can produce creatures previous text-to-3D could not; closest 3D precedent of skeleton-controlled creature generation. **[verified]** — [arXiv 2406.16273](https://arxiv.org/abs/2406.16273)
- Shape2Animal (arXiv 2506.20616, latest version May 2026) **(2026)**: VLM picks animal concepts matching natural silhouettes, SDXL+ControlNet renders them — 2D. **[verified]** — [arXiv 2506.20616](https://arxiv.org/html/2506.20616v4)
- Google Chimera Painter (2020): conditional GAN turning part-segmentation maps of combined creatures into images; training outlines derived from 3D creature models; designed for a card game of hybrid creatures — early example of part-segmentation-driven chimeras. **[verified]** — [Google Research blog](https://research.google/blog/using-gans-to-create-fantastical-creatures/)
- PartGen (CVPR 2025): part-level 3D generation/reconstruction with multi-view diffusion — parts meaningful and independently manipulable. **[verified]** — [CVF](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_PartGen_Part-level_3D_Generation_and_Reconstruction_with_Multi-view_Diffusion_Models_CVPR_2025_paper.pdf)
- CoPart (ICCV 2025): generates multiple contextual part latents *simultaneously* with mutual guidance for coherence. **[verified]** — [CVF](https://openaccess.thecvf.com/content/ICCV2025/papers/Dong_From_One_to_More_Contextual_Part_Latents_for_3D_Generation_ICCV_2025_paper.pdf)
- ArtiLatent (arXiv 2510.21432, 2025) **(2025)**: joint part geometry + articulation latent with latent diffusion. **[verified]** — [arXiv 2510.21432](https://arxiv.org/html/2510.21432)
- ComboVerse (2024): compositional 3D generation by combining objects with spatially-aware score distillation. **[from prior knowledge, URL not re-fetched]** — [arXiv 2403.12409](https://arxiv.org/abs/2403.12409)
- Evaluation practice in blending work: human preference studies (Olearo 2025: 100 participants; ATIH: 95 participants) and balance of DINO image similarity / CLIP text similarity to each source (ATIH). **[verified]** — [arXiv 2506.23630](https://arxiv.org/abs/2506.23630); [ATIH](https://xzr52.github.io/ATIH/)

### Inferences
- The coherence lesson common to Chimera, CoPart and DreamCreature: parts must be generated *jointly by one model with shared context*, not generated separately and stitched. The project's planned step (3) — one generator growing flesh over a single blended skeleton — is consistent with that lesson, whereas the previous subtree grafting is the "stitch" regime.
- A project-specific coherence evaluation could combine: (a) parent-share fidelity — a pretrained species/family classifier's soft output on the child should track the weights (the project already has a family-accuracy probe); (b) ATIH-style balance (similarity of child to each parent embedding vs. weight); (c) physical plausibility — support polygon/centre-of-mass check for "stands"; (d) surface continuity — fraction of points near the skeleton-junction regions, or connected-component count of a reconstructed surface (patchwork shows seams/gaps); (e) small user study.

### Gaps
- No 2025–2026 *3D* weighted-hybrid creature paper found; "3D hybrid animals", "Creative 3D generation", "CreativeSynth" were not located in this session.
- No standard "hybrid coherence" metric exists in the sources found; PartEval formulation not detailed on the abstract page.

---

## Q5. Sampling random-but-appealing mixtures; exploration/creativity UIs

### Takeaway
The main literature is interactive evolutionary computation (Picbreeder: users select parents to breed the next generation; collaboration counters fatigue) and its 2026 VLM-driven replication; no paper found that specifically studies Dirichlet weight sampling for appealing blends — concept-distance results (Olearo 2025) suggest parent choice matters more than weights.

### Cited Findings
- Picbreeder (Secretan et al.): users evolve CPPN images by selecting appealing ones as parents of the next generation; key feature is branching from others' results; single users fatigue within ~10–20 generations; the encoding constrains step size and discoveries hinge on serendipity. **[verified]** — [UCF STARS record](https://stars.library.ucf.edu/facultybib2010/1881); [Lehman et al. anecdotes arXiv 1803.03453](https://arxiv.org/pdf/1803.03453)
- "In Search of the Ingredients of Open-Endedness: Replicating Picbreeder with Large Vision-Language Models" (arXiv 2605.23908, 2026) **(2026)**: replaces human selectors with VLMs; reportedly VLMs fall somewhat short of humans (secondhand). **[verified title; result secondhand]** — [arXiv 2605.23908](https://arxiv.org/pdf/2605.23908)
- Concept blending quality depends strongly on conceptual distance between the two concepts and on the seed (Olearo et al. 2025). **[verified]** — [arXiv 2506.23630](https://arxiv.org/abs/2506.23630)

### Inferences
- Random mode: sample weights from a Dirichlet with α≈1–2 (avoids near-degenerate 99/1 splits that look like the parent) and reject splits with min weight < ~0.15; pick parents with a diversity-aware rule (e.g. from different families but within a max "skeleton distance" from Q1's matching cost so the match is meaningful) — this follows from the conceptual-distance finding but is not itself validated in the literature.
- A Picbreeder-style "breed again" UI (child becomes a parent) is cheap given z and skeleton blending are closed under mixing.

### Gaps
- No source on Dirichlet weight sampling or diversity-aware parent selection for generative blending was found.
- Latent-space exploration interface papers (beyond Picbreeder) were not surveyed due to tool-call budget.
