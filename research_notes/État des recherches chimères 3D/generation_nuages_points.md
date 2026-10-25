# Generative models for 3D point clouds (diffusion and flow matching): state of the art to late 2026, for a skeleton-conditioned creature generator

Notation: items marked **[PK]** come from prior knowledge. They were not re-fetched in this session (budget limit). The URLs are canonical arXiv pages, but the numbers should be checked before they are quoted as fact. Items without the mark were read in this session, from the paper page or abstract or from the paper's HTML or PDF.

## 1. Landscape of point cloud generative models and benchmarks

### Takeaway
On ShapeNet (2048 points, chair/airplane/car), 1-NNA has saturated near 53–68%. LION (2022) is still the reference that 2025–2026 papers tie with or narrowly beat; TokenFlow (Oct 2026) and DiPT (Nov 2025) are the newest. The two trends that matter in 2025–2026 are (a) **two-level models**: a small latent set of shape tokens plus a conditional point flow (TokenFlow, Hierarchical FM, OT-MeanFlow3D), and (b) **OT-aware flow matching** that cuts the number of sampling steps (Not-So-OT, OT-MeanFlow3D, min-STP). None of them is skeleton-conditioned. Their training budgets (4×A100 for 4 days, 4×A800 for 4000 epochs, 32×H100) are far above a consumer GPU.

### Cited Findings
**Benchmark protocol and caveats**
- 1-NNA tends toward 50% when the generated and reference sets come from the same distribution. DiPT argues that CD-based metrics are flawed and proposes Density-aware CD (DCD) together with a new **Surface Normal Concordance (SNC)** metric, which uses PCA normals over 20-point neighbourhoods — [DiPT, arXiv 2511.05308](https://arxiv.org/html/2511.05308v1)
- No single 2026 leaderboard ranks all methods. Each paper's tables have to be compared by hand, and the protocols differ (normalisation, CD vs DCD) — [search synthesis; DiPT](https://arxiv.org/html/2511.05308v1)

**Classic lineage [PK]**
- PointFlow (2019): continuous normalizing flow over points plus a shape latent (VAE). This introduced the MMD/COV/1-NNA protocol — [arXiv 1906.12320](https://arxiv.org/abs/1906.12320) [PK]
- ShapeGF (2020): learns the gradient field of the log-density of the surface and samples any number of points by Langevin dynamics. It is an early resolution-free method — [arXiv 2008.06520](https://arxiv.org/abs/2008.06520) [PK]
- DPM (Luo & Hu, CVPR 2021): DDPM over points, conditioned on a shape latent, with per-point independent denoising — [arXiv 2103.01458](https://arxiv.org/abs/2103.01458) [PK]
- PVD (2021): point-voxel diffusion with a PVCNN backbone. Generation and completion, 1000 steps — [arXiv 2104.03670](https://arxiv.org/pdf/2104.03670)
- LION (NeurIPS 2022): hierarchical VAE with a global shape latent plus a point-structured latent, and diffusion in both latent spaces — [arXiv 2210.06978](https://arxiv.org/abs/2210.06978) [PK for the URL]. Its goals were "high generation quality, flexibility for manipulation (conditional synthesis, shape interpolation), smooth surfaces/meshes" — [PSF/LION search](https://ar5iv.labs.arxiv.org/html/2212.01747)
- Point-E (OpenAI 2022): text→image→point cloud with a transformer diffusion model, 1K points plus an upsampler to 4K — [arXiv 2212.08751](https://arxiv.org/abs/2212.08751) [PK]
- PSF / "Fast Point Cloud Generation with Straight Flows" (CVPR 2023, Wu et al.): rectified-flow straightening (reflow) plus distillation to **one step**, under 0.1 s per generation, "comparable to standard diffusion" — [CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Fast_Point_Cloud_Generation_With_Straight_Flows_CVPR_2023_paper.pdf), [arXiv 2212.01747](https://arxiv.org/pdf/2212.01747)

**Transformers on points**
- DiT-3D (NeurIPS 2023): DiT on voxelized point clouds with 3D patch embedding and 3D window attention, then devoxelization. It can be fine-tuned from ImageNet DiT-2D. It cuts 1-NNA by 4.59 points and raises COV by 3.51 versus the previous SOTA (CD) — [arXiv 2307.01831](https://arxiv.org/abs/2307.01831)
- FastDiT-3D (ECCV 2024): masks about **99%** of voxel tokens during training and reaches better 1-NNA/COV at 128³ voxels for **6.5%** of DiT-3D's training cost. Decoder-depth ablation: depth 4 = 11 h, 1-NNA-CD 50.09; depth 2 = 7 h, 51.26 — [ECCV paper](https://ecva.net/papers/eccv_2024/papers_ECCV/papers/11278.pdf), [supp](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11278-supp.pdf), [arXiv 2312.07231](https://arxiv.org/abs/2312.07231v1)
- DiPT, Diffusion Point Transformer (arXiv Nov 2025): DiT-3D-style diffusion with a **Point Transformer V3** backbone. Points are serialized along space-filling curves (Z-order, Hilbert), with no voxelization, plus Serialized Patch Attention (patch sizes 256/512/1024), xCPE positional encoding and AdaLN. Small config: 12 blocks, width 384, 6 heads, 2048 points. Training: **32×H100, 10,000 epochs**, DDPM 1000 steps. 1-NNA DCD/EMD: Chair 68.68/64.47, Airplane 63.70/74.32, Car 61.11/60.26. DiT-3D gets 99.00 1-NNA-DCD on Chair under its protocol — [arXiv 2511.05308](https://arxiv.org/html/2511.05308v1)
- Spectral representations (arXiv 2609.08253, Sept 2026): a DiT-3D S/4 baseline on Chair goes from 1-NNA 0.850/0.875 to 0.583/0.627 **after 5K iterations**, i.e. much faster convergence — [arXiv 2609.08253](https://arxiv.org/pdf/2609.08253) (snippet only, PDF too large to fetch)

**Flow matching / OT works (2025–2026, the most recent SOTA)**
- **Not-So-Optimal Transport Flows (ICLR 2025, NVIDIA)**: Table 3 at 1000 steps, 1-NNA CD/EMD. Ours: Chair 0.5551/0.5763, Airplane 0.6864/0.6185, Car 0.5966/0.5355. LION: Chair 0.5370/0.5234, Airplane 0.6741/0.6123, Car 0.5341/0.5114. PVD: Chair 0.5626/0.5332. PSF: Chair 0.5612/0.5642. DiT-3D: Chair 0.6072/0.5604. The authors call themselves "comparable to PVD, slightly worse than LION" at 1000 steps, but best at **10–20 steps**. Completion works in 5 steps where others need about 50. Backbone PVCNN. Training: 600K iterations, batch 256, **about 4 days on 4×A100** — [arXiv 2502.12456](https://arxiv.org/html/2502.12456v1), [project page](https://research.nvidia.com/labs/genair/not-so-ot-flow/)
- **OT-MeanFlow3D (arXiv 2509.22592, v4 Oct 2026)**: MeanFlow (average velocity, single-step generation) plus OT. Exact point-level OT is computed **inside each conditioned shape**, while shape-level transport happens in a compact latent space, which avoids nested OT. It claims much lower training and inference cost than diffusion and other flow models. The abstract gives no figures — [arXiv 2509.22592](https://arxiv.org/abs/2509.22592)
- **Hierarchical Flow Matching (arXiv 2608.05557, Aug 2026)**: latent FM for the global shape plus conditional point FM. Both use OT-FM with MSE losses and need as few as **15 Euler steps per flow**. The velocity net is a point-wise MLP, and the authors list a transformer velocity net as future work. The abstract gives no numbers — [arXiv 2608.05557](https://arxiv.org/abs/2608.05557)
- **TokenFlow (arXiv 2610.27418, Oct 2026, the newest)**: M=64 latent tokens of width 128 from a "Context Bottleneck Aggregation" encoder (cross-attention), as a diagonal Gaussian with KL to N(0,I). Prior flow: a 4-layer transformer over tokens, with minibatch OT on the tokens only. Point flow: a per-point MLP of width 128 plus cross-attention to the tokens plus AdaLN, with independent noise (no point-level OT), 20+20 Euler steps, 2048 points. 1-NNA CD/EMD: Airplane 66.65/60.55, Car 53.44/50.87, Chair 54.26/52.21 vs LION 67.41/61.23, 53.41/51.14, 53.70/52.34. Training: **4×A800, 4000 epochs, batch 128**. Token ablation, 1-NNA-CD Airplane: M=1 81.59, M=8 76.57, M=32 72.30, **M=64 66.65**, M=128 70.03 — [arXiv 2610.27418](https://arxiv.org/html/2610.27418)

**Latent-set (VecSet) models for meshes/SDF**
- 3DShape2VecSet (SIGGRAPH 2023): encodes a dense point cloud into a fixed-size **set of latent vectors** by cross-attention from learnable queries or FPS-subsampled points. The decoder is cross-attention from query 3D points to the set, predicting occupancy/SDF. It is designed to work with transformer diffusion — [arXiv 2301.11445](https://arxiv.org/pdf/2301.11445), [GitHub](https://github.com/1zb/3DShape2VecSet)
- TripoSG is a VecSet-based image-to-3D model, reused as a backbone by later work — [MeshReGen arXiv 2604.28134](https://arxiv.org/pdf/2604.28134)
- "Representing 3D Shapes with 64 Latent Vectors" (ICCV 2025): shows that a VecSet of only 64 latents can be enough for 3D diffusion — [arXiv 2503.08737](https://arxiv.org/pdf/2503.08737) (title only)
- MSVS-VAE (2026): multi-scale anchored VecSet — [arXiv 2607.24436](https://arxiv.org/pdf/2607.24436) (title only)

### Inferences
- The user's model (Point-DiT plus per-shape exact OT, 10 Euler steps) already sits on the 2025–2026 frontier recipe: OT-FM plus few steps plus a transformer. Hierarchical FM explicitly names a transformer velocity net as future work.
- 1-NNA differences of 1–3 points between LION, TokenFlow and Not-So-OT are within protocol noise. For a portfolio, surface error and qualitative detail matter more than leaderboard rank.
- The TokenFlow ablation (M=64 best, M=128 worse) suggests a shape latent of a few dozen tokens of width about 128 is enough for ShapeNet-scale shapes.

### Gaps
- Not verified this session: TIGER, PDT (Patch Diffusion Transformer), FrePolad, PointFM, Make-a-Shape, CLAY, Dora (sharp-edge sampling), TripoSG token counts. No benchmark numbers retrieved for them.
- DiT-3D per-category numbers and its point count were not retrieved (abstract only).
- No Papers-with-Code leaderboard was consulted (the site was discontinued in 2025 and redirects to HF; not verified).

## 2. Per-shape (point-level) OT coupling: methods, costs and quality vs exact matching

### Takeaway
Within-shape point matching is the method recommended in the 2025–2026 literature. Minibatch OT across shapes barely helps on point clouds (about 6% cost reduction even at batch 256), while equivariant/per-shape matching gives 48.7% at batch 1. The scalable recipe is **Not-So-OT**: precompute OT once per shape on a dense superset (Hungarian up to about 10K points, **Sinkhorn-divergence Wasserstein gradient flow on GPU for 100K points in about 30 s**), then subsample and **perturb the noise (β≈0.2)**. Pure OT (β=0) gave clearly worse 1-NNA than independent coupling.

### Cited Findings
- Minibatch OT "achieves only about 6% reduction in the cost even with batch size 256". Equivariant (permutation-aware) OT gives a **48.7% reduction even at batch size 1**, but costs about 2.2 s per batch at B=1, roughly 40× slower than independent coupling — [Not-So-OT](https://arxiv.org/html/2502.12456v1)
- Hungarian is O(M³) and **takes 220 s at M=10,000**. For M>10K the paper minimizes Sinkhorn divergence by Wasserstein gradient flow with the GPU KeOps/GeomLoss implementation of Feydy et al. 2019: **about 30 s for 100K points**. At training time the cost is "merely additional indexing of a cached array" — [Not-So-OT](https://arxiv.org/html/2502.12456v1)
- Hybrid coupling x0' = √(1−β)·x0 + √β·ε. Chair, 100 steps, 1-NNA CD/EMD: β=0 **0.7289/0.6647**; β=0.01 0.6223/0.6156; β=0.1 0.5968/0.5884; **β=0.2 0.5853/0.5741**; β=0.5 0.5861/0.5780; β=1 (independent) 0.5989/0.5869 — [Not-So-OT](https://arxiv.org/html/2502.12456v1)
- Superset size (Chair, β=0.2): M=2048 (no superset, same as the user's set-up) 0.6352/0.6254 vs **M=100K 0.5725/0.5627**. Subsampling a 2048-point training set from a larger matched superset clearly helps — [Not-So-OT](https://arxiv.org/html/2502.12456v1)
- Equivariant flow matching (Klein, Krämer, Noé, NeurIPS 2023): a symmetry-aware OT cost, minimized over permutations (Hungarian) and rotations (Kabsch), inside minibatch OT. It gives near-optimal paths with small batches — [NeurIPS 2023 poster](https://neurips.cc/virtual/2023/poster/70968), [FLOWR description, arXiv 2504.10564](https://arxiv.org/pdf/2504.10564); related: [Equivariant FM with hybrid probability transport, arXiv 2312.07168](https://arxiv.org/pdf/2312.07168), [EquiFlow arXiv 2412.11082](https://arxiv.org/pdf/2412.11082)
- OT-MeanFlow3D runs exact point-level OT within each conditioned shape and puts shape-level transport in latent space. This decoupling is the same structure the user plans (skeleton or z condition plus per-shape point matching) — [arXiv 2509.22592](https://arxiv.org/abs/2509.22592)
- TokenFlow uses minibatch OT only on its 64 latent tokens and **independent** point noise, yet still matches LION — [arXiv 2610.27418](https://arxiv.org/html/2610.27418)
- min-Sliced Transport Plans (arXiv 2511.19741, v4 May 2026): a learned 1D "slicer" gives a closed-form 1D OT plan, and the slicer transfers across distribution pairs, supporting **amortized** point cloud alignment and flow-based generation (used within an OT-MeanFlow framework). No timings appear in the abstract — [arXiv 2511.19741](https://arxiv.org/abs/2511.19741), [search snippet](https://arxiv.org/pdf/2511.19741)
- GPU log-domain Sinkhorn CUDA library: **12× faster than POT at n=m=8192**, stable down to ε=1e-4, with 3D point cloud matching among its applications — [ResearchGate preprint](https://www.researchgate.net/publication/404427861_Fast_Log-Domain_Sinkhorn_Optimal_Transport_with_Warp-Level_GPU_Reductions)
- Sinkhorn can be slower than linear programming when the entropic regularization is very small — [Springer, partial transport for registration](https://link.springer.com/article/10.1007/s43670-025-00097-1); sliced optimal partial transport for point clouds (CVPR 2023) compares wall-clock time with Sinkhorn/LP — [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Bai_Sliced_Optimal_Partial_Transport_CVPR_2023_paper.pdf)
- PUFM++ approximates OT (EMD) with an **auction algorithm** in a pre-alignment stage — [PUFM++ summary](https://lacuna.tiptreesystems.com/work/pufm-point-cloud-upsampling-via-enhanced-flow-matching/wrk_9a98c85039df7e0b44795ee54637c62c), [arXiv 2512.20988](https://arxiv.org/html/2512.20988v1)
- OT-CFM / minibatch OT (Tong et al. 2023) is the general framework. Mousavi-Hosseini et al. (arXiv 2509.25519) study semidiscrete and Sinkhorn couplings for flow matching in general (not point-specific) — [survey, arXiv 2606.04092](https://arxiv.org/pdf/2606.04092)

### Inferences
- The user's "2 precomputed draws as good as 8" agrees with Not-So-OT's finding that a fixed precomputed map plus a small perturbation is enough. The noise diversity comes from β, not from many draws.
- **Recommended for 40K creatures:** precompute per shape offline, once, on GPU, with a GeomLoss Sinkhorn gradient flow on a 16K–100K superset (about 30 s per shape at 100K, so 40K shapes ≈ 14 GPU-days at 100K; about 1–3 s per shape at 16K is my extrapolation, not measured). Then subsample 2048–4096 matched pairs per iteration plus β≈0.1–0.2 noise. This replaces online OT entirely, and it also gives resolution-free training, because any subset of the superset is matched.
- Fully online alternatives, ranked by cost: (1) Hilbert/Morton-curve sorting of both noise and data, which is O(N log N) and an approximation (the quality gap vs Hungarian is not measured in any found source); (2) GPU Sinkhorn at N=4096 (milliseconds with KeOps; my estimate); (3) learned slicers (min-STP).
- The user's β=0 corresponds to the configuration that was worst for diversity in Not-So-OT (1-NNA 0.73 vs 0.585). Adding β≈0.2 is a cheap experiment, and it may also help the "surface too thick" symptom by regularizing the flow (hypothesis).

### Gaps
- No paper found that measures quality of Hilbert-curve/sorted coupling vs exact Hungarian for point clouds.
- No published per-shape cost of GPU Sinkhorn at 2K–16K points inside a training loop; the figures above are extrapolations.
- The "Song et al. 2024" equivariant OT flow is cited in Not-So-OT, but its exact title was not verified.

## 3. Resolution-free / variable point count generation, upsampling and local refinement

### Takeaway
Three working patterns exist: (1) **latent read/write (Perceiver-style) denoisers** à la PointInfinity, trained at low resolution and sampled up to 131K points, where *more* test points *improve* quality; (2) **per-point decoders conditioned on a global code**, as in DPM, ShapeGF, TokenFlow and Hierarchical FM, which are trivially resolution-free because points are denoised independently given the latent; (3) **conditional upsamplers**, as in Point-E, PUDM and PUFM/PUFM++ (flow matching from sparse to dense in local patches, about 5 steps). Each fits a planned "global pass then dense face pass".

### Cited Findings
- PointInfinity (CVPR 2024, Meta FAIR): a fixed-size latent "read/write" transformer. Cost is linear in the number of points because points interact only through the latent. Trained at low resolution, it generates up to **131K points on CO3D (31× Point-E)**. Raising the test resolution *improves* fidelity: CD 0.227 at 1024 points → **0.151 at 8192**. When the read module is restricted to the training token count, the gain disappears. The authors also show the resolution acts like CFG, trading fidelity against variability — [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_PointInfinity_Resolution-Invariant_Point_Diffusion_Models_CVPR_2024_paper.html), [arXiv 2404.03566](https://arxiv.org/pdf/2404.03566)
- TokenFlow's encoder is a "context bottleneck" of 64 tokens with cross-attention, so the point flow is a per-point MLP plus cross-attention to tokens, and its cost is **linear in N** — [arXiv 2610.27418](https://arxiv.org/html/2610.27418)
- DiPT avoids O(N²) attention by serializing points along space-filling curves and attending within patches of 256–1024 points (the PTv3 approach) — [arXiv 2511.05308](https://arxiv.org/html/2511.05308v1)
- DiT-3D uses 3D window attention to control token count; FastDiT-3D masks about 99% of tokens — [DiT-3D](https://arxiv.org/abs/2307.01831), [FastDiT-3D](https://ecva.net/papers/eccv_2024/papers_ECCV/papers/11278.pdf)
- Not-So-OT trains on 2048-point subsets of a 100K superset, so the matched coupling is itself resolution-agnostic — [arXiv 2502.12456](https://arxiv.org/html/2502.12456v1)
- PUDM: a conditional DDPM for upsampling, with the sparse cloud as condition and a rate factor for **arbitrary scales**. It converges slowly — [arXiv 2312.02719](https://arxiv.org/html/2312.02719v1)
- PUFM (AAAI 2026): flow matching *from the sparse cloud* (after midpoint interpolation) directly to dense, with EMD pre-alignment. Fewer steps than diffusion upsamplers; generalizes to ScanNet/KITTI — [arXiv 2501.15286](https://arxiv.org/html/2501.15286v1), [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/37685)
- PUFM++ (arXiv 2512.20988): **arbitrary-ratio** upsampling, flow matching in **local patch space**, about **5 inference steps** — [arXiv 2512.20988](https://arxiv.org/html/2512.20988v1)
- Point-E used a separate upsampler diffusion model (1K→4K) conditioned on the coarse cloud — [arXiv 2212.08751](https://arxiv.org/abs/2212.08751) [PK]

### Inferences
- For "fast global pass in the browser plus dense face on demand", the most coherent design is the following. (a) Global pass: the current Point-DiT at 2048–4096 points. (b) Face pass: a second, small conditional flow (PUFM++ style, local patches) that starts from the coarse face points plus interpolated jitter (not pure noise), crops a region around the head joints, and is conditioned on the coarse face points, head joint features and z. Starting from the coarse points instead of Gaussian noise makes 3–5 Euler steps realistic, according to PUFM/PUFM++.
- Alternatively, make the global model resolution-free: train on random subsets of a large matched superset (Not-So-OT) with cross-attention or latent bottlenecks. A user can then simply sample more points in the head region. Two ways to do this are drawing extra noise points near the head joints (noise points matched to head-region surface points during training), or PointInfinity-style extra test points.
- Full self-attention over points is the bottleneck: 4096 tokens is fine on GPU, but in a browser (WebGPU/ONNX) 8K+ tokens needs a latent bottleneck (PointInfinity/TokenFlow) or serialized local attention (PTv3/DiPT).

### Gaps
- No paper found doing explicitly "local refinement of a selected region (e.g. a face)" for point generation. PUFM++'s patch-based design is the closest.
- No browser/WebGPU latency numbers found for any point diffusion model.

## 4. Shape latent codes alongside a structural condition (skeleton): VAE latents, smooth interpolation, preventing bypass, multi-condition CFG

### Takeaway
Recent two-level point flows (TokenFlow, Hierarchical FM, OT-MeanFlow3D, LION) all use a **KL-regularized latent set of a few dozen tokens**, encoded by cross-attention, and decode points with a flow conditioned on it. A second "prior" flow models the latent so that generation without an encoder works. For the user's case (skeleton plus optional z), the standard tools are: a KL-weighted (β-VAE) bottleneck to keep z small and smooth; **independent condition dropout** so that z-free, skeleton-only and joint modes exist; and **InstructPix2Pix-style compositional CFG** with separate scales.

### Cited Findings
- TokenFlow: 64 tokens × 128, diagonal Gaussian posterior via the reparameterization trick, KL to N(0,I) summed over tokens, an entropy term, all loss weights = 1. A token-level prior flow with minibatch OT lets z be sampled at generation time. Interpolation and extrapolation are shown qualitatively only. Its latents are also good features (ModelNet10 linear SVM 87.9%, ModelNet40 93.9%). No dropout is reported — [arXiv 2610.27418](https://arxiv.org/html/2610.27418)
- Hierarchical FM: a compact "structured" latent space modelled by its own latent flow, then a conditional point flow — [arXiv 2608.05557](https://arxiv.org/abs/2608.05557)
- OT-MeanFlow3D: shape-level transport in a compact latent space, point-level OT within each shape — [arXiv 2509.22592](https://arxiv.org/abs/2509.22592)
- Not-So-OT for completion: a 256-d latent from a PVCNN encoder conditions the flow (the LION recipe) — [arXiv 2502.12456](https://arxiv.org/html/2502.12456v1)
- 3DShape2VecSet: learnable-query or FPS-initialized latent set built by cross-attention, used as diffusion target — [arXiv 2301.11445](https://arxiv.org/pdf/2301.11445)
- InstructPix2Pix: CFG over **two conditions** with separate scales s_I and s_T. The guided estimate = uncond + s_I·(img − uncond) + s_T·(img+text − img), which costs three forward passes per step. Diffusers defaults: guidance 7.5, image guidance 1.5 — [arXiv 2211.09800](https://arxiv.org/pdf/2211.09800), [Diffusers docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/pix2pix.md). Training-time dropout (from the paper, [PK], not confirmed in the fetched snippets): 5% image only, 5% text only, 5% both.
- MoViE distils both guidance terms into one network pass — [arXiv 2412.06578](https://arxiv.org/pdf/2412.06578)
- PointInfinity: test-time resolution acts like a CFG knob for the fidelity/diversity trade-off — [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_PointInfinity_Resolution-Invariant_Point_Diffusion_Models_CVPR_2024_paper.html)

### Inferences
- **Anti-bypass recipe** (synthesis, not from one paper):
  1. z is a small KL-regularized token set, e.g. 8–32 tokens × 64–128, with a KL weight tuned so that z carries what the skeleton does not (surface style, proportions, mass).
  2. Encode z from the shape *after* removing what the skeleton determines, or feed the encoder only a random subset of the points.
  3. Drop z with probability about 10–20% and the skeleton about 5–10%, independently, so that z=∅ (skeleton-only), skel=∅ and both exist.
  4. Measure bypass directly: swap z between creatures with the same skeleton, and check that family accuracy under skeleton-only sampling matches the current 90–92%.
- **Smooth interpolation:** a KL-regularized Gaussian latent plus a learned prior flow over z (TokenFlow, Hierarchical FM). Interpolate in the prior's noise space or linearly in z, and keep the same point noise x0 across the interpolation, which is easy with a fixed OT-matched noise.
- **CFG with two conditions:** use InstructPix2Pix-style s_skel and s_z. Given the earlier measurement (OT+CFG only 55–60% family accuracy), CFG and OT may interact badly. Test CFG scales in 1.0–2.0 only, or use guidance only on z.

### Gaps
- No point-cloud paper found that studies explicitly "latent bypassing a structural condition", or that reports KL-weight sweeps for point VAEs. The anti-bypass recipe above is inferred from general practice.
- The InstructPix2Pix 5% dropout figure was not confirmed in fetched text.

## 5. Fast-training tricks for diffusion/flow transformers and what fits on one consumer GPU

### Takeaway
The largest documented speedups are **representation alignment** (REPA: >17.5× on SiT for images; spectral or alignment-style regularizers on DiT-3D converge in about 5K iterations), **token masking** (MaskDiT about 30% of DiT's cost; FastDiT-3D **6.5%** of DiT-3D's cost with 99% voxel masking), **logit-normal t-sampling** (SD3), and **few-step OT flows** (Not-So-OT: 10–20 steps; MeanFlow: 1 step). Published point cloud SOTA used 4×A100–32×H100. Nothing found reports a consumer-GPU point-cloud result, so a single RTX 4060 is unexplored territory. Masking and alignment are the tricks most likely to transfer.

### Cited Findings
- REPA (ICLR 2025 oral): aligns noisy hidden states with clean features from a frozen pretrained encoder (DINOv2), giving **>17.5× faster SiT training**. It matches SiT-XL at 7M steps in under 400K steps, FID 1.42 with CFG — [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d9e42b4d7163931f3689d6d6fbaa11d0-Abstract-Conference.html), [arXiv 2410.06940](https://export.arxiv.org/pdf/2410.06940)
- Logit-normal timestep sampling (SD3, Esser et al. 2024) beats uniform rectified flow, EDM and LDM-linear. It concentrates mass at mid t — [arXiv 2403.03206](https://arxiv.org/pdf/2403.03206), [Improved noise schedule, arXiv 2407.03297](https://arxiv.org/pdf/2407.03297). Some later models add a resolution-dependent shift plus a 10% uniform mixture — [Nucleus-Image arXiv 2604.12163](https://arxiv.org/pdf/2604.12163)
- MaskDiT (TMLR 2024): masking 50% of patches halves the per-iteration cost. FID 2.28 for **273 h on 8×A100 = 31% of DiT's time**, 29% at 512². Unmasked fine-tuning adds under 6% — [arXiv 2306.09305](https://arxiv.org/pdf/2306.09305), [GitHub](https://github.com/Anima-Lab/MaskDiT)
- FastDiT-3D: about 99% masking of voxelized point clouds, **6.5% of the training cost**, 7–11 h ablation runs — [ECCV 2024](https://ecva.net/papers/eccv_2024/papers_ECCV/papers/11278.pdf), [supp](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11278-supp.pdf)
- Spectral representations on DiT-3D: Chair 1-NNA 0.850→0.583 (CD) after only 5K iterations — [arXiv 2609.08253](https://arxiv.org/pdf/2609.08253)
- Reference training budgets: Not-So-OT 600K iterations × batch 256, 4 days on 4×A100; TokenFlow 4000 epochs, 4×A800; DiPT 10,000 epochs, 32×H100 — [Not-So-OT](https://arxiv.org/html/2502.12456v1), [TokenFlow](https://arxiv.org/html/2610.27418), [DiPT](https://arxiv.org/html/2511.05308v1)
- Few-step sampling: PSF 1 step (<0.1 s); Not-So-OT best at 10–20 steps; Hierarchical FM 15 Euler steps per flow; TokenFlow 20+20; PUFM++ about 5 — sources above

### Inferences
- Applying these to the user's model, in order of expected value per effort:
  1. **Logit-normal t** (one line).
  2. **β-perturbed OT noise from a large matched superset** (sections 2–3).
  3. **Random point dropping or masking** (train on 50% of the 2048–4096 points with full-set fine-tuning at the end). This works naturally with sets, and in my opinion it is cheaper than MaskDiT's decoder, because a point set can simply be subsampled.
  4. **REPA-style alignment** to a frozen 3D encoder (e.g. a pretrained Point Transformer/PointBERT, or the user's own z encoder) on an intermediate block. The point-cloud transfer is unverified.
- At 40K shapes on one 8 GB RTX 4060 with bf16, a 10M-parameter Point-DiT at 2048 points is feasible. Multiplying epochs × dataset size as in the published runs (4000 epochs) is not. Expect to rely on OT straightness and few epochs (a few hundred on 40K shapes), using free Kaggle/Colab GPUs (T4/P100, no bf16 on T4) for longer runs.

### Gaps
- No 2025–2026 point-cloud paper found that reports training on a single consumer GPU or that applies REPA to points.
- MDT (Masked Diffusion Transformer, Gao et al.) numbers not retrieved.
- No source on bf16/mixed-precision effects specific to point diffusion.
