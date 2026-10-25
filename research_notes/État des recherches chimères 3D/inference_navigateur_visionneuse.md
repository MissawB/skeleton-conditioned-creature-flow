# In-browser generative inference and 3D point-cloud viewers (state as of October 2026)

Scope: whether a flow-matching Point-DiT (~10–30M params, 6 blocks, width 256, 8 heads, self-attention over N points plus cross-attention to ≤100 joint tokens, 10 Euler steps) can generate 2048–4096 points in 1–2 s in a laptop browser for free, plus how to view and ship the results. Research date: 2026-10-25. The tool budget was limited (about 17 calls), so several runtimes are listed under Gaps rather than covered.

## Q1. Runtimes and browser support (WebGPU status, operator coverage, fp16)

### Takeaway
As of October 2026, WebGPU is on by default in Chrome/Edge (desktop and most Android devices), Safari 26 (macOS/iOS/iPadOS) and Firefox (Windows since 141, Apple Silicon macOS since 145/147). Firefox on Linux and Android, and some Linux Chrome setups, still lack it, so a WASM fallback is still needed. ONNX Runtime Web (directly, or through Transformers.js v4, which uses the same ORT-built C++ WebGPU runtime) is the most mature general-purpose route. It covers all the ops the Point-DiT needs (LayerNorm, GELU, MatMul, Softmax, Where, Gather*), but the JSEP operator table does not list TopK, and it notes that masks are not implemented for its fused contrib `Attention`/`MultiHeadAttention`.

### Cited Findings
**Browser WebGPU status (gpuweb implementation-status wiki, last edited 2 Oct 2026):**
- Chromium (Chrome/Edge): Mac, Windows x86/x64 and ChromeOS supported since 113. Android 12+ on ARM/Qualcomm/Intel GPUs since 121, Imagination GPUs on Android 16+ since 139, Samsung Xclipse "TBD". Linux: Intel Gen12+ since 144, NVIDIA (driver ≥535.183.01) on Wayland since 147, other Linux setups behind a flag. Windows ARM64 behind a flag (`--enable-unsafe-webgpu`) — [gpuweb wiki](https://github.com/gpuweb/gpuweb/wiki/implementation-status)
- Firefox (wgpu): shipped on Windows in 141 (blog post dated 15 Jul 2025). On Apple Silicon Mac it shipped in 145 for macOS 26+ and in 147 for all macOS versions. Intel Macs and Linux: Nightly only, with Linux "expected 2026". Android: behind a flag (`gfx.webgpu.ignore-blocklist`), "expected 2026" — [gpuweb wiki](https://github.com/gpuweb/gpuweb/wiki/implementation-status)
- Safari 26: WebGPU enabled by default on macOS, iOS/iPadOS and visionOS — [gpuweb wiki](https://github.com/gpuweb/gpuweb/wiki/implementation-status); also [Wikipedia: WebGPU](https://en.wikipedia.org/wiki/WebGPU)
- Different revisions of web.dev's "WebGPU is now supported in major browsers" post give different Chrome minimum versions (113 vs 144, probably because Linux was added later). Treat the gpuweb wiki as authoritative — [web.dev](https://web.dev/blog/webgpu-supported-major-browsers)
- WebGPU fp16 (`shader-f16`) has been available in Chrome/Edge since 121 — [Microsoft Open Source blog, Feb 2024](https://opensource.microsoft.com/blog/2024/02/29/onnx-runtime-web-unleashes-generative-ai-in-the-browser-using-webgpu/)

**ONNX Runtime Web / Transformers.js:**
- Transformers.js v4.0.0 (March 2026) uses a new WebGPU runtime "completely rewritten in C++", built with the ONNX Runtime team and tested across about 200 architectures. The same code runs in the browser, Node, Bun and Deno. `env.useWasmCache` caches the WASM runtime files for offline use — [transformers.js releases](https://github.com/huggingface/transformers.js/releases); [releases.sh summary](https://releases.sh/hugging-face/transformers-js/highlights)
- Transformers.js v4.3 adds WebGPU support for Safari 26+ and an ONNX Runtime upgrade — [newreleases v4.3.0](https://newreleases.io/project/github/huggingface/transformers.js/release/4.3.0)
- Transformers.js v4 vendor claims: BERT embedding models up to 4x faster; GPT-OSS 20B (q4f16) at about 60 tok/s on an M4 Pro Max — [transformers.js releases](https://github.com/huggingface/transformers.js/releases); [howaiworks summary](https://howaiworks.ai/blog/transformers-js-v4-release) (secondary)
- ORT WebGPU EP: FlashAttention was added for MultiHeadAttention with QKV bias, which previously fell back to the unfused path. Reported gain on whisper-tiny-int4: about 92 → 134 tok/s (+45%). The change appears in v1.27.0 and in plugin-ep-webgpu v0.2.1 — [commit (semanticdiff)](https://app.semanticdiff.com/gh/microsoft/onnxruntime/commit/3b007a6873c33591d6f527c78928852a0df609b9); [v1.27.0 notes](https://newreleases.io/project/github/microsoft/onnxruntime/release/v1.27.0); [plugin-ep-webgpu v0.2.1](https://newreleases.io/project/github/microsoft/onnxruntime/release/plugin-ep-webgpu%2Fv0.2.1)
- ORT v1.29 and v1.30 add WebGPU PagedAttention and tuned FlashAttention (about 2× faster decode, about 1.3× faster varlen prefill). This work targets LLMs — [v1.29.0](https://newreleases.io/project/github/microsoft/onnxruntime/release/v1.29.0); [v1.30.0](https://newreleases.io/project/github/microsoft/onnxruntime/release/v1.30.0); [commit](https://app.semanticdiff.com/gh/microsoft/onnxruntime/commit/ebd7bad168218df30eb3224000f6d75c69c11f3a)
- ORT v1.31.0 adds more WebGPU kernels (GatedRMSNorm activation, hyper-connection ops) — [v1.31.0](https://newreleases.io/project/github/microsoft/onnxruntime/release/v1.31.0)
- ORT Web README (up to v1.30) still calls WebGPU support "experimental" and links to `webgpu-operators.md` — [onnxruntime-web 1.30.0 README](https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/README.md)
- WebGPU operator table (`js/web/docs/webgpu-operators.md`, auto-generated, JSEP path):
  - Supported: LayerNormalization (17+), Gelu (ai.onnx 20+ and com.microsoft), FastGelu, SkipLayerNormalization, MatMul, Softmax, Where, GatherND, GatherElements, Einsum, ReduceMean, CumSum, Expand, GroupQueryAttention.
  - MultiHeadAttention and Attention (com.microsoft) are listed with the note "need implementing mask and past/present".
  - **TopK is not listed** — [webgpu-operators.md](https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md)

### Inferences
- The Point-DiT uses only standard ops (Linear/MatMul, LayerNorm, GELU, Softmax, SiLU, Mul/Add for AdaLN-Zero). All of them have WebGPU kernels in ORT Web, so ORT Web (raw `onnxruntime-web`, without Transformers.js pipelines) is the lowest-risk runtime. Transformers.js adds nothing for a custom architecture beyond its loader and caching.
- **Masks:** the fused contrib `MultiHeadAttention` does not implement masks in JSEP. If the variable skeleton size (≤100 joints) is handled with a padding mask on cross-attention, either (a) let the exporter emit plain MatMul + Add(mask) + Softmax, which is unfused but cheap for 100 keys, or (b) export one graph per joint count (bucketing) without a mask. Self-attention over the points needs no mask if N is fixed per session.
- **kNN/TopK:** if the face pass or any post-processing needs kNN/TopK, do it outside the ONNX graph (in JS, or a small custom WGSL kernel). With the JSEP EP an unsupported op would fall back to the CPU and force GPU↔CPU copies.
- Firefox on Linux and Android and older Android devices will hit the WASM EP. For ~2–4K tokens × 10 steps that will be slow (see Q2), so the page should detect `navigator.gpu` and switch to playing precomputed trajectories, or use the server fallback.

### Gaps
- Not researched within the budget: TensorFlow.js WebGPU, WebLLM/MLC (TVM Unity), tinygrad WebGPU, Burn/wgpu→WASM. No sourced 2026 maturity data for them is in these notes. WebLLM is LLM-specific and unlikely to suit a custom DiT. Burn/wgpu and hand-written WGSL are viable but require reimplementing the model.
- The operator table found is the JSEP (TypeScript) one. I did not confirm whether the newer native C++ WebGPU EP (used by Transformers.js v4) has a different coverage list or mask support for its FlashAttention path.
- Safari's and Firefox's `shader-f16` and subgroups support was not confirmed.

## Q2. Measured browser performance, and estimated cost for a 2048–8192-token attention model

### Takeaway
Published in-browser numbers are mostly vendor best cases: SD-Turbo in under 1 s on an RTX 4090, SAM encoder 19× faster on WebGPU than WASM on an RTX 3060 laptop, MiniLM embeddings at about 24 ms on WebGPU vs about 386 ms on WASM. I found no independent measurement of a diffusion-style transformer on an integrated laptop GPU. A back-of-envelope FLOP count puts a 10-step generation at ~0.5 TFLOP for N=2048 and ~1.5 TFLOP for N=4096. That makes 1–2 s plausible for N=2048 on recent integrated GPUs and comfortable on discrete GPUs, while N=4096 is borderline on integrated GPUs unless attention is fused and fp16.

### Cited Findings
- ORT Web WebGPU: Segment Anything encoder 19× faster and decoder 3.8× faster than WASM, measured on a laptop with an RTX 3060 and Core i9. SD-Turbo generates an image "within one second" on an RTX 4090. An end-to-end example is in onnxruntime-inference-examples `js/sd-turbo` — [Microsoft Open Source blog, 29 Feb 2024](https://opensource.microsoft.com/blog/2024/02/29/onnx-runtime-web-unleashes-generative-ai-in-the-browser-using-webgpu/)
- The web-txt2img package lists SD-Turbo 512×512 as WebGPU-only, about a 2.3 GB download, and gives no timings — [web-txt2img README](https://cdn.jsdelivr.net/npm/web-txt2img@0.3.1/README.md)
- Transformers.js community WebGPU benchmark (all-MiniLM-L6-v2, Chrome 122/124, pre-v4): batch 1 takes 386 ms on WASM vs 23.8 ms on WebGPU — [Xenova webgpu-embedding-benchmark discussion](https://huggingface.co/spaces/Xenova/webgpu-embedding-benchmark/discussions/1)
- ORT Web WebGPU MultiHeadAttention has a FlashAttention path (see Q1), and decode/prefill FlashAttention programs were tuned in v1.29/1.30 — [v1.30.0](https://newreleases.io/project/github/microsoft/onnxruntime/release/v1.30.0)
- An unverified claim exists of Gemma 4 at 255 tok/s in the browser with AI-written WebGPU kernels. A single secondary source; do not rely on it — [pasqualepillitteri.it](https://pasqualepillitteri.it/de/news/7152/gemma-4-255-tok-im-browser-webgpu-fable-5)

### Inferences (own calculations, not sourced; check them by benchmarking)
- **FLOPs per block at d=256, N points** (×2 for multiply-add):
  - QKV projection: 3·N·d²·2.
  - QKᵀ plus AV: 2·N²·d·2.
  - Output projection: N·d²·2.
  - MLP (4×): 16·N·d².
  - Cross-attention to ≤100 tokens: about 4·N·d² + tiny terms.
- **N=2048:** about 8 GFLOP per block, ~48 GFLOP per network call, **~0.5 TFLOP for 10 Euler steps**.
- **N=4096:** about 25 GFLOP per block (attention N² terms ≈17 GFLOP dominate), ~150 GFLOP per call, **~1.5 TFLOP for 10 steps**.
- **N=8192:** attention alone is about 69 GFLOP per block, so roughly 5 TFLOP or more for 10 steps. That is a multi-second job even on discrete GPUs, so it is fine for an on-demand "face pass" but not interactive.
- **Memory of unfused attention:** the score matrix for 8 heads × N² in fp32 is about 134 MB at N=2048 and about 537 MB at N=4096. That is risky on integrated GPUs and mobile. FlashAttention-style fused MHA in ORT WebGPU, or fp16, or chunking queries in the graph, avoids it.
- **Assumed effective WebGPU throughput:** a few hundred GFLOP/s on a modern integrated GPU (Intel Xe / AMD 780M / Apple M-series base), several TFLOP/s on a discrete laptop GPU. On that assumption:
  - N=2048 takes about 0.5–2 s on integrated GPUs and well under 0.5 s on discrete GPUs.
  - N=4096 takes about 2–6 s on integrated GPUs.
  - These are assumptions, not sourced numbers.
- Stated more simply, the 1–2 s target is realistic for **2048 points with fp16 weights and fused attention**. Options for the "global 4096" case:
  - Generate 2048 points and upsample cheaply (e.g. a second copy of noise points conditioned on the same skeleton, run as a batch).
  - Reduce the steps.
  - Accept about 3 s on integrated GPUs.
- **CPU/WASM fallback:** the MiniLM ratio (~16×) suggests WASM would be one to two orders of magnitude slower. 0.5–1.5 TFLOP on WASM SIMD with multithreading would take tens of seconds, so treat it as unusable for live generation.
- **Overhead:** keep tensors on the GPU between the 10 Euler steps. Use ORT `preferredOutputLocation: 'gpu-buffer'` / IO binding, or fold the Euler loop into the graph, to avoid 10 GPU→CPU readbacks. I did not confirm the exact API name for current versions; see Gaps.

### Gaps
- I found no independent laptop or integrated-GPU benchmark for SD-Turbo or any DiT in the browser, no Whisper/ViT numbers from 2026, and no measured per-layer WebGPU attention timings at N=2048–8192. Run a microbenchmark of the exported model before committing.
- I could not confirm whether the 2024 blog's "IOBinding planned" became the current GPU-buffer output API in ORT Web 1.2x/1.3x.

## Q3. Exporting PyTorch to ONNX for the web: dynamic shapes, masks, quantisation, size and caching

### Takeaway
Use `torch.onnx.export(..., dynamo=True)` (the default exporter in recent PyTorch) with `dynamic_shapes`, not `dynamic_axes`. Keep the model a single file well under 2 GB; at ~10–30M params it is 20–60 MB in fp16 (fp32 roughly double), far below every browser and ORT limit. ORT Web's limits are 2 GB per protobuf, a ~2 GB ArrayBuffer in Chrome, and a 4 GB maximum model.

### Cited Findings
- The `torch.export`-based exporter (`dynamo=True`) is "the recommended and default way to export models to ONNX". `dynamic_shapes` is meant for `dynamo=True`, and `dynamic_axes` for the legacy `dynamo=False` path — [PyTorch ONNX docs (stable)](https://docs.pytorch.org/docs/stable/onnx); [2.13 docs](https://docs.pytorch.org/docs/2.13/onnx.html)
- ModelOpt's ONNX exporter has a `scaled_dot_product_attention` translation that accepts an optional `attn_mask` (it is not the native ONNX `Attention` op) — [NVIDIA ModelOpt export_onnx](https://nvidia.github.io/Model-Optimizer/pr-preview/pr-2321/reference/generated/modelopt.torch.quantization.export_onnx.html)
- ORT Web size limits: a protobuf can be at most 2 GB, so larger models use external data. Chrome's maximum ArrayBuffer is 0x7fe00000 bytes (~2 GB); ORT works around it with `WebAssembly.Memory`, but such buffers are not transferable and do not work with the Proxy feature. **There is currently no way for ORT Web to run models larger than 4 GB** (WASM 32-bit addressing) — [ORT Web: Working with Large Models](https://onnxruntime.ai/docs/tutorials/web/large-models.html)
- External data has to be passed in session options (`externalData: [{path, data}]`) — [ORT Web deploy docs](https://onnxruntime.ai/docs/tutorials/web/deploy.html)
- A community report says files must each be under 2 GB to be cacheable (browser Cache API) — [HF discussion, Xenova/Phi-3-mini](https://huggingface.co:443/Xenova/Phi-3-mini-4k-instruct/discussions/2)
- Transformers.js v4 caches the WASM runtime (`env.useWasmCache`). Model files are cached by its loader — [transformers.js releases](https://github.com/huggingface/transformers.js/releases)

### Inferences
- **Size:** 30M params is about 60 MB in fp16 or about 30 MB in int8. Load the file once and store it with the Cache API or IndexedDB yourself if you use raw onnxruntime-web.
- **Precision:** prefer fp16 weights and activations on WebGPU over int8. int8 MatMul kernels on WebGPU usually dequantize, so the gain is mostly in download size. For a flow-matching velocity field, int8 errors accumulate over 10 integration steps, so validate the fp16 trajectory against PyTorch, e.g. with Chamfer distance on the final cloud.
- **Dynamic shapes:**
  - Declare N (points) and J (joints) as `torch.export.Dim`. Avoid data-dependent Python control flow, such as `if mask.any()`.
  - Keep AdaLN-Zero time conditioning as an ordinary graph input t of shape [B].
  - Masks: pass a float additive mask of shape [B,1,1,J] rather than a boolean mask through SDPA. Alternatively pad J to a fixed 100 and mask, which is cheap. A static J=100 makes the graph simpler and more robust.
- **Graph optimisation:** run the onnxruntime transformer optimizer or onnxslim offline, check that the attention subgraphs fuse (or deliberately keep them unfused for the masked cross-attention), then `float16` conversion with `keep_io_types=True`.
- **Euler loop:** export only the velocity network and run the 10-step loop in JS with GPU-resident tensors. That keeps the graph simple and allows step-by-step animation of the trajectory, which is the current viewer's feature.

### Gaps
- No source found on whether PyTorch 2.9+/opset 23 emits the native ONNX `Attention` op from SDPA, or on whether ORT WebGPU fuses it. Verify by exporting and inspecting the graph.
- No 2026 source found on browser Cache API per-entry limits beyond the community 2 GB note. This is irrelevant at 60 MB.

## Q4. Live 3D generation demos already running in the browser

### Takeaway
In-browser 3D generation exists but is rare and early. The clearest example is TripoSplat WebGPU, an image-to-Gaussian-splat pipeline on ORT WebGPU, released as an "alpha". I found no public in-browser point-cloud diffusion or flow demo, such as a Point-E port, so a live Point-DiT would be a distinctive portfolio piece.

### Cited Findings
- **TripoSplat WebGPU:** ONNX artifacts run a single-image → Gaussian-splat pipeline in the browser through ONNX Runtime WebGPU, and the image never leaves the device. It is labelled an alpha engineering release, and parity with the official model is not yet confirmed. Live demo at triposplat-webgpu.vercel.app — [HF model card](https://huggingface.co/Yosun/TripoSplat-WebGPU/)
  - It can offload to a community GPU network (MutualGPU) when the local device is weak — [Product Hunt](https://www.producthunt.com/p/triposplat-webgpu/triposplatwebgpu-local-or-free-mutualgpu)
- **Visionary (arXiv 2512.08478, Dec 2025):** a WebGPU Gaussian-splat platform with per-frame ONNX inference, a "Gaussian Generator" plug-in interface and a three.js plug-in — [arXiv](https://arxiv.org/html/2512.08478v1)
- **WebSplatter (arXiv 2602.03207, Feb 2026):** cross-device Gaussian splatting in browsers through WebGPU — [arXiv](https://arxiv.org/html/2602.03207v1)
- **Arrival.Space Splat.js:** an MIT-licensed WebGPU library that runs SfM and splat *training* in a browser tab — [radiancefields.com](https://radiancefields.com/arrival.space-splat.js-trains-gaussian-splats-in-browser)
- **Splat renderers:** Scthe's gaussian-splatting-webgpu (math write-up and demo) — [GitHub](https://github.com/Scthe/gaussian-splatting-webgpu). W3C basic example of splats in WebGPU (2026) — [w3.org](https://www.w3.org/2026/splat/)

### Inferences
- The TripoSplat approach (ORT WebGPU, ONNX artifacts on the Hub, graceful offload) is close to the target architecture and shows that the toolchain works for 3D generative nets in 2026.

### Gaps
- I did not find Point-E or Shap-E browser ports, or any point-cloud flow-matching demo in the browser. This is an absence in search results, not proof that none exist.

## Q5. Point-cloud viewers, rendering, catalogue UI, and Hugging Face hosting limits

### Takeaway
For 10K–100K points, Three.js `Points` (WebGLRenderer) with a custom round-sprite or splat shader is enough and the safest choice. The WebGPU renderer and its compute particles scale to millions, but at least one 2026 report shows it slower than WebGL for draw-heavy scenes. On hosting:
- Static Spaces are still free.
- **Gradio and Docker Spaces now require a paid plan (PRO) to create**, with one exception: free personal accounts can host up to 2 Gradio Spaces on ZeroGPU.
- The "free CPU Gradio Space" fallback assumption is therefore no longer valid for a free account. The fallback becomes ZeroGPU Gradio or precomputed playback.

### Cited Findings
- **Three.js tips (2026):** CPU-side particle updates bottleneck around 50K particles, and WebGPU compute shaders reach millions. Use `instancedArray` for GPU-persistent buffers. A practitioner reports a 500K-particle shader-driven hero section — [utsubo: 100 Three.js tips (2026)](https://www.utsubo.com/blog/threejs-best-practices-100-tips)
- **Three.js forum (r181, M4 Pro, Chrome):** WebGPURenderer was slower than WebGL for many cube meshes. At 5K cubes, WebGL ran at ~350 fps vs ~140 fps on WebGPU; at 50K cubes, 40 fps vs 3–6 fps. This is draw-call-bound and not a test of point sprites — [discourse.threejs.org](https://discourse.threejs.org/t/webgpu-performance-issue/87939)
- **WebGPU compute shader discussion for Three.js** — [discourse.threejs.org](https://discourse.threejs.org/t/webgpu-compute-shaders-support/66758)
- **HF Spaces:**
  - Static Spaces are free for everyone. Gradio and Docker Spaces "run on compute and require a paid plan to create: PRO for personal accounts…". Free personal accounts "can still host up to 2 Gradio Spaces running on ZeroGPU".
  - CPU Basic is 2 vCPU, 16 GB RAM, 50 GB non-persistent disk.
  - Free hardware Spaces sleep after a period of inactivity.
  - Static Spaces can read Space variables in JS through `window.huggingface.variables`.
  - [HF Spaces overview](https://huggingface.co/docs/hub/spaces-overview)
- Forum reports describe 403 "CPU-basic quota limit" errors for free accounts starting Spaces, which fits the new quota policy — [HF forum](https://discuss.huggingface.co/t/403-cpu-basic-quota-limit-on-my-first-and-only-space/178658); [HF forum](https://discuss.huggingface.co/t/question-about-the-cpu-basic-quota-limit/176628)
- **HF repo limits (all repo types, including Spaces):**
  - Fewer than 100K files per repo recommended, hard maximum of 10K files per folder.
  - Files recommended under 200 GB, with a 500 GB hard limit per file.
  - Fewer than 100 files per commit when using the HTTP API.
  - Free accounts get "best-effort" public storage and 100 GB of private storage.
  - Files are served through CloudFront.
  - [HF storage limits](https://huggingface.co/docs/hub/storage-limits)

### Inferences
- **Catalogue of 40K creatures:**
  - Size: ≤100 joints × 3 coordinates (plus parent index) in fp16 is about 0.6–0.8 KB per creature, plus a small latent (e.g. 64 fp16 = 128 B). That is about 30–40 MB raw, compressing well.
  - Packaging: ship it as a few binary shards (e.g. 4–8 files of 5–10 MB) or a Parquet/Arrow file, plus a separate small JSON/TSV index of names, species and families for search. Never 40K small files, given the 10K-per-folder limit and request overhead.
  - Search: load the index eagerly and fetch skeleton shards lazily with HTTP Range or per-shard requests. For the UI, filter the index client-side (e.g. a fuzzy search over ~40K strings is instant) and render the list with virtual scrolling.
- **Rendering:** 2–4K points (or up to 100K for a dense face pass) are trivial for `THREE.Points` with a `BufferGeometry`. Write each Euler step's positions into the position attribute and set `needsUpdate` to animate the trajectory live, or interpolate between stored steps in the vertex shader. For a nicer look, use a screen-space round-sprite shader with depth-based size and simple sphere-impostor shading, or "splat"-style Gaussian sprites with additive or depth-sorted blending. Full 3DGS renderers are unnecessary for a few thousand isotropic points.
- **Share the GPU device:** ORT WebGPU and the Three.js WebGPURenderer can in principle share a GPUDevice for zero-copy rendering, but this is complex. Copying 4096×3 floats back to the CPU once per step costs almost nothing, so keep WebGLRenderer.
- **Fallback chain:**
  1. WebGPU live generation with ORT Web.
  2. If `navigator.gpu` is missing, play precomputed trajectories (the current Space behaviour) and optionally try WASM with a reduced N (e.g. 1024 points, a progress bar).
  3. Optionally a ZeroGPU Gradio Space (free for personal accounts, up to 2) called through `@gradio/client` for the dense face pass.

### Gaps
- No sourced numbers for free CPU Basic Gradio inference speed for a 30M-param DiT, and no documented sleep timeout duration. Both matter less now that creating CPU Gradio Spaces requires PRO.
- I did not find a per-file size limit specific to static Spaces beyond the general repo limits, or potree's 2026 status. Potree targets massive LiDAR clouds and is overkill here.
- No r17x benchmark of `THREE.Points` at 100K points was found. This is widely considered trivial, but not sourced here.
