<div align="right">
  <b>English</b> | <a href="README.zh.md">中文</a>
</div>

# awesome-china-gpu

Tracking the progress of domestic Chinese GPU/AI chip adaptation for LLM inference (vLLM, SGLang) and related open-source repositories.

When enterprises migrate LLM inference from NVIDIA to domestic chips, they commonly face four challenges: **can't run, runs slow, runs unstable, runs inaccurate**. This repo continuously tracks the support status, official repositories, open-source projects, and developer resources for vendors including Huawei Ascend, Cambricon, MetaX, Biren Technology, Moore Threads, Hygon, Iluvatar CoreX, Tsingmicro, and Innosilicon — helping algorithm engineers and technical decision-makers quickly evaluate the feasibility of domestic computing power.

---

## Table of Contents

- [Background: The "Four Walls" of Domestic Computing Power](#background-the-four-walls-of-domestic-computing-power)
  - [Wall 1: Ecosystem → Can't Run](#wall-1-ecosystem--cant-run)
  - [Wall 2: Memory → Runs Slow](#wall-2-memory--runs-slow)
  - [Wall 3: Communication → Runs Unstable](#wall-3-communication--runs-unstable)
  - [Wall 4: Precision → Runs Inaccurate](#wall-4-precision--runs-inaccurate)
  - [Strategies to Cross the Four Walls](#strategies-to-cross-the-four-walls)
- [Domestic GPU Inference Engine Comparison](#domestic-gpu-inference-engine-comparison)
- [DeepSeek Model Support Matrix](#deepseek-model-support-matrix)
- [Vendor Details](#vendor-details)
  - [Huawei Ascend](#huawei-ascend)
  - [Cambricon](#cambricon)
  - [MetaX](#metax)
  - [Biren Technology](#biren-technology)
  - [Moore Threads](#moore-threads)
  - [Hygon](#hygon)
  - [Iluvatar CoreX](#iluvatar-corex)
  - [Tsingmicro](#tsingmicro)
  - [Innosilicon](#innosilicon)
- [Contributing](#contributing)
- [Disclaimer](#disclaimer)

---

## Background: The "Four Walls" of Domestic Computing Power

> Reference: [SiliconFlow: Four Challenges of Domestic Computing Power for LLM Inference](https://industry.caijing.com.cn/20260326/5149782.shtml) (Chinese, adapted)

By 2026, inference computing demand accounts for over 70% of total AI workloads, and domestic chip market share has exceeded 41%. However, theoretical computing power ≠ business performance — migrating LLM inference from NVIDIA to domestic chips faces four invisible but formidable barriers.

### Wall 1: Ecosystem → Can't Run

**This is the first hurdle encountered.** Mainstream open-source communities (HuggingFace), inference frameworks (vLLM, SGLang), and optimization tools (DeepSpeed, FlashAttention) are all built on the CUDA ecosystem.

**CUDA Code Tight Coupling:**
Critical inference operators like PagedAttention and FlashAttention are implemented directly as CUDA kernels. FlashAttention is specifically optimized for NVIDIA GPU SRAM size and bank architecture. Domestic chips have their own software stacks (Huawei CANN, Cambricon NeuWare, Moore Threads MUSA, Hygon DTK). Simply porting open-source code causes tiling strategy failures and performance degradation. While automated translation tools exist (e.g., CUDA-to-CANN), the generated code performs far worse than native hand-written implementations.

**Missing Operator "Last Mile":**
When new techniques like MoE's Grouped GEMM or novel activation functions emerge, NVIDIA provides timely support in cuDNN. Domestic chip operator libraries lag behind. When encountering unsupported operators, systems may fall back to Device-to-Host Copy, fragmenting the inference pipeline and spiking latency.

### Wall 2: Memory → Runs Slow

**LLM inference is fundamentally memory-bound, not compute-bound.** Think of it this way: a world-class chef can prepare 100 dishes per second, but the ingredient porter can only move 10 dishes' worth of ingredients per second — the bottleneck is the porter, not the chef.

**Bandwidth Bottleneck:**
NVIDIA H100/A100 HBM bandwidth exceeds 3 TB/s. Due to supply chain and advanced packaging (CoWoS) limitations, some domestic chips have significantly lower HBM bandwidth. During the decode phase, compute units often wait for data — the lower the memory bandwidth, the lower the tokens per second (TPS).

**Capacity Bottleneck:**
Early domestic single-card memory was typically small (32G/64G). Models like 72B/671B cannot fit on a single card and must be sharded across multiple cards. Once multi-card sharding is required, data must be transferred frequently between cards, causing inference latency to spike.

### Wall 3: Communication → Runs Unstable

**With multi-card parallelism, inter-card communication speed determines inference service viability.** At this point, overall system performance is determined not by single-card compute or memory access, but by the "communication efficiency" between cards.

**Interconnect Bandwidth Gap:**
NVIDIA NVLink/NVSwitch achieves up to 900 GB/s bidirectional bandwidth. Domestic chips typically rely on PCIe Gen5 (~128 GB/s) or proprietary interconnect protocols (e.g., Huawei HCCS), with a significant bandwidth gap. During Tensor Parallel (TP) inference, communication time may exceed computation time, creating a "adding cards doesn't speed up" dilemma.

**Communication Library Maturity:**
NVIDIA's NCCL has matured over years of iteration. Domestic communication libraries (e.g., HCCL, CNCL) may still experience communication jitter, deadlocks, packet loss, and connection failures under extreme concurrency, posing risks for production online services.

### Wall 4: Precision → Runs Inaccurate

**This is the dividing line between "works" and "works well."** After crossing the first three walls, the system runs at acceptable speed, but output quality may reveal new problems.

**FlashAttention Deep Adaptation Gap:**
FlashAttention requires fine-grained control over GPU SRAM (on-chip cache). Different domestic chips have completely different SRAM sizes and architectures from NVIDIA. Blindly applying the algorithm without chip-specific tiling parameter tuning can result in performance degradation.

**Precision Gaps and INT4 Absence:**
NVIDIA natively supports BF16/FP8/INT4. Some domestic chips primarily support FP16, with incomplete BF16 hardware acceleration. Inference requires cast operations, risking precision overflow that causes garbled output or service crashes. W4A16 (4-bit weights + 16-bit activations) is the current industry trend (e.g., Marlin Kernel), but most domestic chips are still improving INT4 hardware acceleration.

### Strategies to Cross the Four Walls

- **Match scenarios, start pragmatically:** For latency-insensitive tasks with slower model iteration, domestic computing power is already cost-effective. Gain experience through targeted entry points rather than seeking one-shot solutions.
- **Invest in talent, build depth:** The true moat is having teams that can perform low-level kernel optimization and distributed communication.
- **Embrace collaboration:** Work with chip vendors and MaaS platforms to solve adaptation challenges together, creating a positive feedback loop.

---

## Domestic GPU Inference Engine Comparison

> **Note:** As of June 2026, SGLang's official native hardware support includes Huawei Ascend NPU and Innosilicon Fantasy series. Other vendors currently support vLLM or their own inference engines. This table is continuously updated.

| Vendor | Core Platform | SGLang | vLLM | Custom Engine | Precision Support | Product Line |
|--------|--------------|--------|------|---------------|-------------------|--------------|
| **Huawei Ascend** | CANN 9.0.0 | [✅ Official](https://docs.sglang.ai/docs/hardware-platforms/ascend-npus/ascend_npu) | [✅ v0.18.0](https://github.com/vllm-project/vllm-ascend) | MindIE | FP16/BF16/INT4 | 910B / 910C / 950 |
| **Cambricon** | NeuWare / BANG | ❌ N/A | [✅ v0.11.2-dev](https://github.com/Cambricon/vllm-mlu) | MagicMind | FP16/INT8 | SiYuan 590 / 370 |
| **MetaX** | MXMACA | ❌ N/A | [✅ v0.20.0](https://github.com/MetaX-MACA/vLLM-metax) | — | FP16/BF16/INT8 | XiYun C500 |
| **Biren** | BIRENSUPA | ❌ N/A | ❌ No public repo | — | FP16/BF16 | BiLi 166 |
| **Moore Threads** | MUSA | ❌ N/A | [✅ v0.22.0-dev](https://github.com/MooreThreads/vllm-musa) | — | FP16/BF16/INT8 | MTT S series |
| **Hygon** | DTK (ROCm) | ❌ N/A | [✅ v0.8.5 image](https://developer.sourcefind.cn/servicelist/detail?post_id=61036870-b3c7-11f0-9989-acde48001122&active=TagDownload) | — | FP16/BF16/FP32 | ShenSuan K100 / Z100 |
| **Iluvatar CoreX** | IXUCA SDK | ❌ N/A | [✅ DeepSparkInference](https://github.com/Deep-Spark/DeepSparkInference) | IxRT + IGIE | FP16/INT8/INT4 | TianGai 150 / ZhiKai 100 |
| **Tsingmicro** ⚠️ | RAISA / FlagOS | ❌ N/A | ⚠️ [vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL) (indirect) | FlagScale | FP16/BF16 | TX81 |
| **Innosilicon** | FTUCA | [✅ Official site](https://www.fantasyxpu.com/product/gr308smax) | [✅ Official site](https://www.fantasyxpu.com/product/gr308smax) | FantRT | TF32/FP16/BF16/INT8 | GR308S MAX / GR308S |

### Hardware Specs Comparison

| Vendor | Model | Memory | Bandwidth | Interconnect | FP16 Compute | Highlights |
|--------|-------|--------|-----------|-------------|-------------|------------|
| **Ascend** | 910C | 64GB HBM | ~1.5 TB/s | HCCS | ~320 TFLOPS | PD disagg, most mature ecosystem |
| **Cambricon** | SiYuan 590 | 48GB HBM | ~1.2 TB/s | CNCL | 345 TFLOPS | Best value, 5D hybrid parallel |
| **MetaX** | XiYun C500 | 48GB HBM | ~1.3 TB/s | PCIe 5.0 | ~300 TFLOPS | High CUDA compatibility |
| **Biren** | BiLi 166 | 64GB HBM | ~1.5 TB/s | BILink | ~360 TFLOPS | National security certification |
| **Moore Threads** | MTT S series | 32GB GDDR6 | ~700 GB/s | PCIe 5.0 | ~200 TFLOPS | Full-function GPU, community-driven |
| **Hygon** | ShenSuan K100 | 64GB HBM | ~1.0 TB/s | PCIe | ~200 TFLOPS | ROCm ecosystem compatible |
| **Iluvatar CoreX** | TianGai 150 | 48GB HBM | ~1.2 TB/s | PCIe 4.0 | ~280 TFLOPS | TVM-based custom engine |
| **Tsingmicro** | TX81 | — | — | TSM-LINK | — | CGRA dataflow architecture |
| **Innosilicon** | GR308S MAX | 112GB GDDR6 | 2240 GB/s | PCIe 5.0 | — | SGLang+vLLM dual support |

---

## DeepSeek Model Support Matrix

DeepSeek-V3.1/R1/V4 series MoE models require advanced parallelism strategies (Expert Parallelism, etc.), making them an important benchmark for domestic GPU inference capability.

| Vendor | DeepSeek-V3.1 | DeepSeek-R1 Distilled | DeepSeek-V4 | Notes |
|--------|--------------|----------------------|-------------|-------|
| **Ascend** | ✅ SGLang / vLLM | ✅ | ✅ | CANN 9.0.0 + DeepEP lib, PD disagg |
| **Cambricon** | ✅ vllm-mlu | ✅ | ✅ **day0** | Day-0 support announced 2026/4/24, MLU 370+ |
| **MetaX** | ✅ vLLM-metax | ✅ | — | Official team active |
| **Moore Threads** | ✅ vllm-musa | ⚠️ Verifying | — | MUSA community-driven |
| **Iluvatar CoreX** | ✅ DeepSparkInference | ✅ Full R1 series | — | IXUCA 4.4.0 verified |
| **Innosilicon** | ✅ Listed on official site | ✅ | — | 8× GR308S MAX runs full 671B |
| **Hygon** | ⚠️ Runnable (vLLM) | ⚠️ Runnable | — | 2-card TP verified Qwen3-8B |
| **Biren** | ❌ No public info | ❌ | — | Awaiting official beta |
| **Tsingmicro** | ⚠️ Via FlagOS | ⚠️ | — | CGRA architecture, indirect support |

---

## Vendor Details

### Huawei Ascend

The leading domestic AI chip vendor with full-stack self-developed stack (hardware + CANN + MindIE inference engine). Official support for both SGLang and vLLM makes it the most complete ecosystem among domestic chips.

**Latest Updates (Jun 2026):**
- vLLM-Ascend latest stable [v0.18.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.18.0) (Apr 2026), aligned with vLLM v0.18.0
- SGLang Ascend support [v0.5.10-npu](https://docs.sglang.ai/docs/hardware-platforms/ascend-npus/ascend_npu), CANN 9.0.0 + HDK 25.5.2
- Supports PD disaggregation (MemFabric-Hybrid KV cache transfer), multimodal models (Qwen3-VL, etc.)
- Provides [DeepEP compatibility library](https://github.com/sgl-project/sgl-kernel-npu) as a drop-in for DeepSeek-ai DeepEP
- Triton on Ascend (triton-ascend 3.2.1) available

| Resource | Link |
|----------|------|
| Developer Community | https://www.hiascend.com/developer |
| CANN Docs | https://www.hiascend.com/cann |
| vLLM-Ascend | https://github.com/vllm-project/vllm-ascend |
| vLLM Ascend Docs | https://docs.vllm.ai/projects/ascend/en/latest/ |
| SGLang Ascend Docs | https://docs.sglang.ai/docs/hardware-platforms/ascend-npus/ascend_npu |
| MindIE Inference Engine | https://www.hiascend.com/software/mindie |
| ModelArts (Cloud) | https://www.huaweicloud.com/product/modelarts.html |
| Dev Space (Free Resources) | https://developer.huaweicloud.com/develop/aigallery/notebook |

- Free resources: 100 card-hours/person or 180 hours via Huawei Dev Space
- CANN differs significantly from CUDA; software migration has a learning curve
- Hardware roadmap: 910B → 910C (H100 competitor) → 950 series (expected Q4 2026)

### Cambricon

Strong in inference workloads. The SiYuan series performs well on vLLM, supporting TP/PP/SP/DP/EP 5D hybrid parallelism and PD disaggregation.

**Latest Updates (Apr–Jun 2026):**
- vllm-mlu [v0.11.2-dev](https://github.com/Cambricon/vllm-mlu) — Day-0 support for **DeepSeek-V4** (announced 2026/4/24)
- Covers Transformer, MoE, multimodal model architectures
- Built on vLLM plugin system, supports Chunk Prefill, Prefix Caching, Spec Decode, Graph Mode

| Resource | Link |
|----------|------|
| Developer Community | https://developer.cambricon.com/ |
| GitHub Organization | https://github.com/Cambricon |
| vLLM-MLU | https://github.com/Cambricon/vllm-mlu |
| MagicMind Inference Engine | https://developer.cambricon.com/sdk/magicmind |

- SiYuan 590: 345 TFLOPS FP16, approaching A100 levels
- Cost-effective: ~60% of Ascend 910B's cost
- Cloud platforms: MatPool, AutoDL have Cambricon instances
- SGLang: No official/community branch available as of Jun 2026

### MetaX

GPGPU approach. The MXMACA software stack is highly CUDA-compatible at the API level, having adapted 6000+ CUDA applications and 500+ AI models.

**Latest Updates (Jun 2026):**
- vLLM-metax latest [v0.20.0](https://github.com/MetaX-MACA/vLLM-metax/releases/tag/v0.20.0) (Jun 2026), aligned with vLLM v0.20.0, supports Gemma4
- Monthly releases tracking upstream vLLM version cadence
- Docker images available for XiYun C500 series

| Resource | Link |
|----------|------|
| Developer Community | https://developer.metax-tech.com/ |
| ModelZoo | https://developer.metax-tech.com/ai-models/ |
| vLLM-MetaX | https://github.com/MetaX-MACA/vLLM-metax |
| vLLM-MetaX Docs | https://vllm-metax.readthedocs.io/en/latest/ |
| Documentation Center | https://developer.metax-tech.com/api/client/document/preview/1222/index.html |
| Gitee AI Platform | https://ai.gitee.com/ |

- Gitee AI provides XiYun C500 instances for zero-cost model deployment
- New registrants and academics can claim free computing credits

### Biren Technology

BiLi series GPGPU focused on high-performance training and inference. The BiLi 166 supports trillion-parameter model training and inference, certified with national security grade I.

| Resource | Link |
|----------|------|
| Developer Community | https://developer.birentech.com/ |
| BIRENSUPA Platform | https://www.birentech.com/product/software/birensupa/ |

- Next-gen BR20X chip expected commercially in 2026
- Ecosystem is relatively closed; no public vLLM/SGLang repositories found yet

### Moore Threads

Full-function GPU approach with MUSA architecture, community-driven. Uses torchada compatibility layer for CUDA → MUSA migration.

**Latest Updates (Jun 2026):**
- vllm-musa [v0.22.0-dev](https://github.com/MooreThreads/vllm-musa) — based on vLLM V1 engine, vLLM plugin system
- Core components mature: torchada (CUDA→MUSA compat layer), MATE (MUSA AI Tensor Engine), torch_musa (PyTorch backend)
- ccache support for accelerated local builds

| Resource | Link |
|----------|------|
| Developer Platform | https://developer.mthreads.com/ |
| Documentation Center | https://docs.mthreads.com/ |
| vLLM-MUSA | https://github.com/MooreThreads/vllm-musa |
| vLLM-MUSA Docs | https://docs.mthreads.com/vllm-musa/vllm-musa-doc-online/intro/ |
| torchada (CUDA compat layer) | https://github.com/MooreThreads/torchada |
| AutoDL Moore Threads Zone | https://www.autodl.com/ |

- AutoDL offers 30-day free trial — lowest barrier to entry
- Full-function GPU approach supports broader scenarios beyond pure AI

---

### Hygon

Hygon DCU (Deep Computing Unit) is based on AMD CDNA architecture. The DTK software stack is built on the ROCm ecosystem, providing a CUDA-like programming environment with low migration costs.

| Resource | Link |
|----------|------|
| Developer Community (GuangYuan) | https://developer.sourcefind.cn/sourcefind |
| Adaptation Guide | https://developer.sourcefind.cn/document/b60fc21f-dfda-11f0-b0a4-0242ac150003 |
| vLLM Image (0.8.5) | https://developer.sourcefind.cn/servicelist/detail?post_id=61036870-b3c7-11f0-9989-acde48001122&active=TagDownload |
| PyTorch Image | https://developer.sourcefind.cn/servicelist/detail?post_id=9f296762-b3c7-11f0-9a0f-acde48001122&active=TagDownload |
| Gitee Repository | https://gitee.com/anolis/hygon-devkit |
| Tutorial (vLLM Deployment) | https://www.psvmc.cn/article/2026-02-05-ai-hygon-llm.html |

- DTK based on ROCm, supports `torch.cuda` API without code changes
- vLLM 0.8.5 official Docker image released, verified with Qwen3-8B dual-card TP
- Production models: ShenSuan K100_AI, Z100, P800
- No SGLang official support; vLLM is the primary inference path
- No public cloud instances yet — hardware purchase required

### Iluvatar CoreX

Iluvatar CoreX's TianGai (training) and ZhiKai (inference) series GPUs use the IXUCA SDK as their core software stack, providing a complete inference toolchain through the DeepSpark open-source community.

**Latest Updates (Apr–Jun 2026):**
- DeepSparkInference verified **50+ LLM models**, covering Qwen3 (including MoE 235B), DeepSeek-V3.1, DeepSeek-R1 full distillation series, ERNIE-4.5, MiniCPM-o-2, and multimodal models
- Baidu PaddlePaddle Hackathon collaboration (Mar–Jun 2026) on TianGai 150 hardware
- SDK updated to IXUCA 4.4.0

| Resource | Link |
|----------|------|
| DeepSpark Community | https://www.deepspark.org.cn |
| DeepSparkInference | https://github.com/Deep-Spark/DeepSparkInference |
| IxRT Open-Source Engine | https://github.com/Deep-Spark/iluvatar-corex-ixrt |
| MoArk (TianGai 150 instances) | https://moark.com |
| Knowledge Base | https://ixkb.iluvatar.com.cn:9443 |

- **vLLM:** DeepSparkInference verified 50+ LLM models including Qwen3, DeepSeek-V3.1/R1, Llama3, ChatGLM, ERNIE-4.5
- **Custom Engines:** IxRT (inference runtime) + IGIE (TVM-based engine), supports INT8/FP16
- **SGLang:** No adaptation yet
- Other frameworks: TGI, LMDeploy, TRT-LLM, FastDeploy
- Free resources: Check Baidu AI Studio hackathon events

### Tsingmicro ⚠️

Tsingmicro uses CGRA (Coarse-Grained Reconfigurable Architecture). Its RPU (Reconfigurable Processing Unit) positions itself as a "third pole" beyond GPUs. The software ecosystem is based on RAISA + Beijing Academy of AI (BAAI) FlagOS open-source framework.

| Resource | Link |
|----------|------|
| Official Site | http://www.tsingmicro.com |
| FlagOS Community | https://github.com/flagos-ai |
| vLLM Plugin (FlagOS) | https://github.com/flagos-ai/vllm-plugin-FL |
| FlagScale (Inference Framework) | https://github.com/flagos-ai/FlagScale |
| FlagGems (Triton Operator Lib) | https://github.com/flagos-ai/FlagGems |

- **Unique Architecture:** Not a GPU — CGRA/RPU dataflow architecture, similar positioning to Groq
- **vLLM:** Indirect support via FlagOS's vllm-plugin-FL, not upstream vLLM
- **SGLang:** No support
- TX81 cloud chip supports trillion-parameter inference; TSM-LINK interconnect scales to 500P
- Completed shareholding reform in 2026, preparing for IPO
- Ecosystem relatively closed; GitHub activity low, primarily dependent on FlagOS community

### Innosilicon

Innosilicon's Fantasy XPU brand offers the Fantasy series GPUs. The FTUCA software stack is fully compatible with CUDA/HIP. The Fantasy 3 GR308S series is one of the few domestic AI chips supporting SGLang.

**Hardware Highlights (GR308S MAX, confirmed via official site):**
- Memory: 112GB GDDR6, bandwidth **2240 GB/s** (highest among domestic cards)
- AI Frameworks: Native support for PyTorch, **vLLM, SGLang**, FantRT
- Programming Models: HIP (CUDA-like), Triton
- Precision: TF32, FP16, BF16, INT8 (all with sparse computation support)
- Interface: PCIe 5.0 x16, SR-IOV virtualization
- Codec: AV1/H.265/H.264 hardware encoding/decoding

| Resource | Link |
|----------|------|
| Fantasy XPU Official | https://www.fantasyxpu.com |
| GR308S MAX Product Page | https://www.fantasyxpu.com/product/gr308smax |
| Compatibility List | https://www.fantasyxpu.com/supportList |
| Driver Download | https://www.fantasyxpu.com/support/driver-download |
| Documentation Center | https://www.fantasyxpu.com/support/docs |

- **SGLang + vLLM dual support:** GR308S MAX product page explicitly lists PyTorch, vLLM, SGLang, FantRT (verified)
- **Large Model Capability:** 8× GR308S MAX (112GB/card) can run full DeepSeek 671B and Qwen3-235B
- Compatible with domestic CPU platforms: Kunpeng, Phytium, Hygon, Zhaoxin, Loongson
- No public GitHub repositories; software distributed via official website

---

## Contributing

PRs welcome for:
- Latest inference engine adaptation updates
- New computing leasing platforms and free resources
- Practical deployment experiences and troubleshooting guides
- Performance benchmark data

---

## Disclaimer

This repo compiles information from publicly available sources. The domestic GPU ecosystem evolves rapidly. Please refer to each vendor's official website for the most current details.
