# awesome-china-gpu

追踪国产 GPU/AI 芯片在 LLM 推理（vLLM、SGLang）上的适配进展与开源仓库。

当企业将大模型推理从英伟达迁移到国产芯片时，普遍面临**跑不通、跑不快、跑不稳、跑不准**四大挑战。本仓库持续收录华为昇腾、寒武纪、沐曦、壁仞、摩尔线程、海光、天数智芯、清微智能、芯动科技等厂商在主流推理框架上的支持情况、官方仓库、开源项目与开发者上手资源，帮助算法工程师和技术决策者快速评估国产算力的落地可行性。

---

## 目录

- [背景：国产算力落地的"四堵墙"](#背景国产算力落地的四堵墙)
  - [第一堵：生态墙 → 跑不通](#第一堵生态墙--跑不通)
  - [第二堵：显存墙 → 跑不快](#第二堵显存墙--跑不快)
  - [第三堵：通信墙 → 跑不稳](#第三堵通信墙--跑不稳)
  - [第四堵：精度墙 → 跑不准](#第四堵精度墙--跑不准)
  - [跨越四堵墙的思路](#跨越四堵墙的思路)
- [国产 GPU 推理引擎对照表](#国产-gpu-推理引擎对照表)
- [厂商详情](#厂商详情)
  - [华为昇腾](#华为昇腾huawei-ascend)
  - [寒武纪](#寒武纪cambricon)
  - [沐曦股份](#沐曦股份metax)
  - [壁仞科技](#壁仞科技biren-technology)
  - [摩尔线程](#摩尔线程moore-threads)
  - [海光信息](#海光信息hygon)
  - [天数智芯](#天数智芯iluvatar-corex)
  - [清微智能](#清微智能tsingmicro)
  - [芯动科技](#芯动科技innosilicon)
- [贡献](#贡献)
- [免责声明](#免责声明)

---

## 背景：国产算力落地的"四堵墙"

> 参考自 [硅基流动：国产算力落地大模型推理的四大挑战](https://industry.caijing.com.cn/20260326/5149782.shtml)，有删改。

2026 年推理算力需求占比已超 70%，国产芯片市占率突破 41%。但理论算力 ≠ 业务性能——将大模型推理从英伟达迁移到国产芯片，中间隔着四堵看不见却异常坚固的高墙。

### 第一堵：生态墙 → 跑不通

**这是所有挑战中最先遇到的门槛。** 主流开源社区（HuggingFace）、推理框架（vLLM、SGLang）和优化工具（DeepSpeed、FlashAttention）都构建在 CUDA 生态之上。

**CUDA 代码强耦合：**
PagedAttention、FlashAttention 等关键推理算子直接写死为 CUDA Kernel。FlashAttention 更是基于 NVIDIA GPU 的 SRAM 大小和 Bank 架构定制的极致优化。国产芯片有自己的软件栈（华为 CANN、寒武纪 NeuWare、摩尔线程 MUSA、海光 DTK），照搬开源代码会导致切块策略失效，速度不升反降。虽然存在自动转译工具（如 CUDA-to-CANN），但生成的代码性能远不如原生手写。企业需要投入大量工程资源进行底层代码重写和调试。

**算子缺失的"最后一公里"：**
大模型技术迭代极快，MoE 的 Grouped GEMM 或新的激活函数出现时，NVIDIA 总能第一时间在 cuDNN 中提供支持。国产芯片算子库更新滞后，遇到不支持的新算子时系统可能选择"回退执行"（Device-to-Host Copy），导致推理流程支离破碎、延迟飙升。

### 第二堵：显存墙 → 跑不快

**大模型推理本质是 Memory-Bound 任务，而非 Compute-Bound。** 打个比方：世界顶级厨师一秒钟能切好 100 盘菜，但食材运输员每秒只能从冷库搬 10 盘菜的原料——瓶颈是运输员的速度，而不是厨师的刀工。

**带宽瓶颈：**
NVIDIA H100/A100 的 HBM 传输速率可达 3TB/s 以上。受限于供应链和先进封装工艺（CoWoS），部分国产芯片的 HBM 带宽明显更低。在模型生成 Token 的解码阶段，计算单元往往在"等数据"，显存带宽越低，每秒生成的 Token 数量（TPS）就越低。

**容量瓶颈：**
早期国产单卡显存普遍较小（32G/64G），面对 72B/671B 大模型单卡无法承载，必须多卡切分。一旦多卡协作，数据必须在卡间频繁传输，计算等通信，推理延迟直线飙升。

### 第三堵：通信墙 → 跑不稳

**多卡并行时，卡间互联速度决定推理服务生死。** 此时决定系统整体性能甚至稳定性的，不再是单卡的计算或访存能力，而是多张卡之间的"沟通效率"。

**互联带宽差距：**
NVIDIA NVLink/NVSwitch 技术实现高达 900GB/s 的双向通信带宽。国产芯片通常依赖 PCIe Gen5（约 128GB/s）或各自的私有互联协议（如华为 HCCS），带宽存在明显差距。在进行 Tensor Parallel（TP）多卡并行推理时，通信耗时可能超过计算耗时，出现"加卡不加速"的窘境。

**通信库的成熟度考验：**
NVIDIA 的集合通信库 NCCL 历经多年迭代，在各种极端负载下表现成熟稳定。国产通信库（如 HCCL、CNCL）在极端并发下仍可能出现通信抖动、死锁、丢包、建链失败等问题，导致推理服务出现莫名超时或波动，支撑核心在线业务存在风险。

### 第四堵：精度墙 → 跑不准

**这是"能用"到"好用"的分水岭。** 跨越前面三堵墙后，系统终于能以可接受的速度稳定运行，但检查输出结果时可能发现新的问题——结果不准。

**FlashAttention 深度适配缺失：**
FlashAttention 依赖对 GPU SRAM（片上缓存）的精细控制，不同国产芯片的 SRAM 大小和架构与 NVIDIA 完全不同。生搬硬套算法逻辑而未针对特定芯片进行 Tiling（切块）参数调优，性能可能不升反降。

**精度缺失与 INT4 缺失：**
英伟达原生支持 BF16/FP8/INT4，部分国产芯片以 FP16 为主，对 BF16 的硬件加速支持不够完善。推理时需要进行 Cast 操作或面临精度溢出风险，导致模型输出乱码甚至服务崩溃。W4A16（4bit 权重 + 16bit 激活）是当前行业趋势（如 Marlin Kernel），多数国产芯片对 INT4 的硬件加速支持尚在完善中。

### 跨越四堵墙的思路

- **场景匹配，务实切入：** 延迟不敏感、模型迭代较慢的任务，国产算力已是极具性价比的选择。找准切入点积累经验，远比一步到位更重要。
- **投资人才，锤炼内功：** 真正的护城河是能深入硬件底层做 Kernel 优化、玩转分布式通信的硬核团队。
- **拥抱协作，共同成长：** 协同芯片厂商、MaaS 平台（如硅基流动的私有化 MaaS 平台），共同解决适配问题，形成正向飞轮。

---

## 国产 GPU 推理引擎对照表

> **说明：** 截至 2026 年 6 月，SGLang 框架官方原生硬件支持列表中，国产芯片包含华为昇腾（Ascend NPU）与芯动科技风华系列。其余厂商目前仅支持 vLLM 或其自研推理引擎。本表将持续更新。

| 企业 | 核心平台 | SGLang 支持 | vLLM 支持 | 自研推理引擎 |
|------|----------|-------------|-----------|-------------|
| **华为昇腾** | CANN | [官方原生支持](https://docs.sglang.io/hardware-platforms/ascend-npus/ascend_npu) | [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend) | MindIE |
| **寒武纪** | Cambricon NeuWare / BANG | ❌ 不支持 | [Cambricon/vllm-mlu](https://github.com/Cambricon/vllm-mlu) | MagicMind |
| **沐曦股份** | MetaX / MXMACA | ❌ 不支持 | [MetaX-MACA/vLLM-metax](https://github.com/MetaX-MACA/vLLM-metax) | — |
| **壁仞科技** | BIRENSUPA | ❌ 不支持 | ❌ 暂无公开仓库 | — |
| **摩尔线程** | MUSA | ❌ 不支持 | [MooreThreads/vllm-musa](https://github.com/MooreThreads/vllm-musa) | — |
| **海光信息** | DTK（基于 ROCm） | ❌ 不支持 | ✅ [官方 Docker 镜像](https://developer.sourcefind.cn/servicelist/detail?post_id=61036870-b3c7-11f0-9989-acde48001122&active=TagDownload) | — |
| **天数智芯** | IXUCA SDK | ❌ 不支持 | [DeepSparkInference](https://github.com/Deep-Spark/DeepSparkInference) | IxRT + IGIE |
| **清微智能** ⚠️ | RAISA / FlagOS | ❌ 不支持 | ⚠️ [vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL)（间接） | FlagScale |
| **芯动科技** | FTUCA | ✅ [风华官网](https://www.fantasyxpu.com) | ✅ [风华官网](https://www.fantasyxpu.com) | FantRT |

---

## 厂商详情

### 华为昇腾（Huawei Ascend）

国产 AI 芯片领军者，全栈自研（硬件 + CANN 软件栈 + MindIE 推理引擎）。在 SGLang 和 vLLM 上均有官方支持，是国产卡中生态最完整的。

| 资源 | 链接 |
|------|------|
| 开发者社区 | https://www.hiascend.com/developer |
| CANN 文档 | https://www.hiascend.com/cann |
| vLLM-Ascend | https://github.com/vllm-project/vllm-ascend |
| vLLM Ascend 文档 | https://docs.vllm.ai/projects/ascend/en/latest/ |
| SGLang 昇腾文档 | https://docs.sglang.io/hardware-platforms/ascend-npus/ascend_npu |
| MindIE 推理引擎 | https://www.hiascend.com/software/mindie |
| ModelArts（云端） | https://www.huaweicloud.com/product/modelarts.html |
| 开发者空间（免费算力） | https://developer.huaweicloud.com/develop/aigallery/notebook |

- 免费算力：华为开发者空间提供 100 卡时/人或 180 小时免费资源
- CANN 与 CUDA 差异较大，软件迁移有一定学习成本
- 硬件路线：910B → 910C（对标 H100）→ 950 系列（预计 2026 Q4 量产）

### 寒武纪（Cambricon）

以推理见长，思元系列在 vLLM 推理框架上表现突出。vLLM 实现支持 TP/PP/SP/DP/EP 5D 混合并行与 PD 分离部署。

| 资源 | 链接 |
|------|------|
| 开发者社区 | https://developer.cambricon.com/ |
| GitHub 组织 | https://github.com/Cambricon |
| vLLM-MLU | https://github.com/Cambricon/vllm-mlu |
| MagicMind 推理引擎 | https://developer.cambricon.com/sdk/magicmind |

- 思元 590 FP16 算力 345 TFLOPS，接近 A100 水平
- 性价比突出，成本约为昇腾 910B 的 60%
- 第三方算力平台：矩池云（MatPool）、AutoDL 有寒武纪实例
- SGLang：截至 2026 年 6 月无官方/社区可用分支

### 沐曦股份（MetaX）

GPGPU 路线，MXMACA 软件栈在 API 层面高度兼容 CUDA，已适配 6000+ CUDA 应用、覆盖 500+ AI 模型，对熟悉英伟达生态的开发者相对友好。

| 资源 | 链接 |
|------|------|
| 开发者社区 | https://developer.metax-tech.com/ |
| 模力方舟（ModelZoo） | https://developer.metax-tech.com/ai-models/ |
| vLLM-MetaX | https://github.com/MetaX-MACA/vLLM-metax |
| 文档中心 | https://developer.metax-tech.com/api/client/document/preview/1222/index.html |
| Gitee AI 平台 | https://ai.gitee.com/ |

- Gitee AI 提供曦云 C500 实例，可零成本部署模型
- 新注册用户/高校师生可领免费算力券

### 壁仞科技（Biren Technology）

壁砺系列 GPGPU，主打大算力训推一体。壁砺 166 系列可支持万亿参数模型训练与推理，已通过国家《安全可靠测评》I 级认证。

| 资源 | 链接 |
|------|------|
| 开发者社区 | https://developer.birentech.com/ |
| BIRENSUPA 平台 | https://www.birentech.com/product/software/birensupa/ |

- 下一代 BR20X 芯片计划 2026 年商用
- 生态较封闭，vLLM/SGLang 公开仓库暂未找到，需关注官方邀测

### 摩尔线程（Moore Threads）

全功能 GPU 路线，MUSA 架构，社区驱动为主。通过 torchada 兼容层实现 CUDA → MUSA 迁移。

| 资源 | 链接 |
|------|------|
| 开发者平台 | https://developer.mthreads.com/ |
| 文档中心 | https://docs.mthreads.com/ |
| vLLM-MUSA | https://github.com/MooreThreads/vllm-musa |
| vLLM-MUSA 文档 | https://docs.mthreads.com/vllm-musa/vllm-musa-doc-online/intro/ |
| AutoDL 摩尔线程专区 | https://www.autodl.com/ |

- AutoDL 上提供 30 天免费试用，门槛最低，适合立刻上手
- 路线更接近"全功能 GPU"，场景更广，但纯 AI 训练场景需全面评估

---

### 海光信息（Hygon）

海光 DCU（Deep Computing Unit）基于 AMD CDNA 架构设计，DTK 软件栈基于 ROCm 生态开发，提供类 CUDA 编程环境，迁移成本低。

| 资源 | 链接 |
|------|------|
| 开发者社区（光源） | https://developer.sourcefind.cn/sourcefind |
| 适配说明文档 | https://developer.sourcefind.cn/document/b60fc21f-dfda-11f0-b0a4-0242ac150003 |
| vLLM 镜像（0.8.5） | https://developer.sourcefind.cn/servicelist/detail?post_id=61036870-b3c7-11f0-9989-acde48001122&active=TagDownload |
| PyTorch 镜像 | https://developer.sourcefind.cn/servicelist/detail?post_id=9f296762-b3c7-11f0-9a0f-acde48001122&active=TagDownload |
| Gitee 代码仓库 | https://gitee.com/anolis/hygon-devkit |
| 实战教程（vLLM 部署 LLM） | https://www.psvmc.cn/article/2026-02-05-ai-hygon-llm.html |

- DTK 基于 ROCm，支持 `torch.cuda` 接口无需修改代码
- vLLM 0.8.5 官方 Docker 镜像已发布，已验证 Qwen3-8B 双卡 TP 推理
- 生产型号：深算 K100_AI、Z100、P800 等
- 无 SGLang 官方支持，vLLM 是主要推理链路
- 免费算力：暂无公开云实例，需采购硬件

### 天数智芯（Iluvatar CoreX）

天数智芯旗下天垓（训练）、智铠（推理）系列 GPU，软件栈以 IXUCA SDK 为核心，通过 DeepSpark 开源社区提供完整推理工具链。

| 资源 | 链接 |
|------|------|
| DeepSpark 社区 | https://www.deepspark.org.cn |
| DeepSparkInference（推理模型库） | https://github.com/Deep-Spark/DeepSparkInference |
| IxRT 开源引擎 | https://github.com/Deep-Spark/iluvatar-corex-ixrt |
| 模力方舟（天垓150 实例） | https://moark.com |
| 知识库 | https://ixkb.iluvatar.com.cn:9443 |

- **vLLM：** DeepSparkInference 已验证 40+ 模型，覆盖 Qwen3、DeepSeek-V3.1、Llama3、ChatGLM 全系列
- **自研引擎：** IxRT（推理运行时）+ IGIE（TVM 引擎），支持 INT8/FP16
- **SGLang：** 暂无适配
- 其他框架适配：TGI、LMDeploy、TRT-LLM、FastDeploy
- 免费算力：关注百度 AI Studio 黑客松活动领取

### 清微智能（Tsingmicro）⚠️

清微智能采用 CGRA（粗粒度可重构架构），其 RPU（可重构处理单元）定位为 GPU 之外的"第三极"。软件生态基于 RAISA + 智源研究院 FlagOS 开源体系。

| 资源 | 链接 |
|------|------|
| 官方主页 | http://www.tsingmicro.com |
| FlagOS 开源社区 | https://github.com/flagos-ai |
| vLLM 插件（FlagOS） | https://github.com/flagos-ai/vllm-plugin-FL |
| FlagScale（推理框架） | https://github.com/flagos-ai/FlagScale |
| FlagGems（Triton 算子库） | https://github.com/flagos-ai/FlagGems |

- **架构特殊：** 不是 GPU，是 CGRA/RPU 数据流架构，定位与 Groq 类似
- **vLLM：** 通过 FlagOS 的 vllm-plugin-FL 间接支持，非直接上游 vLLM
- **SGLang：** 暂无支持
- 云芯片 TX81 支持万亿参数模型推理，TSM-LINK 互联可扩展至 500P
- 2026 年完成股改，筹备 IPO
- 生态较封闭，GitHub 官方账号活跃度低，主要依赖 FlagOS 社区

### 芯动科技（Innosilicon）

芯动科技旗下风华创智（Fantasy XPU）推出风华系列 GPU，FTUCA 软件栈全面兼容 CUDA/HIP。风华 3 号 GR308S 系列是少数支持 SGLang 的国产 AI 芯片之一。

| 资源 | 链接 |
|------|------|
| 风华创智官网 | https://www.fantasyxpu.com |
| 兼容性列表 | https://www.fantasyxpu.com/supportList |
| 驱动下载 | https://www.fantasyxpu.com/support/driver-download |
| 文档中心 | https://www.fantasyxpu.com/support/docs |

- **SGLang + vLLM 双支持：** GR308S MAX 产品规格明确列出 PyTorch、vLLM、SGLang、FantRT
- **大模型能力：** 单机八卡 GR308S MAX（112GB/卡）可运行满血版 DeepSeek 671B 和 Qwen235B
- **自研引擎 FantRT：** 芯动自研高性能推理引擎
- 支持精度：TF32、FP16、BF16、INT8（均支持稀疏计算）
- 支持 SR-IOV 虚拟化、AV1/H.265/H.264 编解码
- 适配国产 CPU 平台：鲲鹏、飞腾、海光、兆芯、龙芯
- 目前无公开 GitHub 仓库，软件通过官网分发

---

## 贡献

欢迎提交 PR 补充以下内容：
- 厂商最新推理引擎适配动态
- 新的算力租赁平台与免费资源
- 实战部署经验 / 踩坑指南
- 性能对比 benchmark 数据

---

## 免责声明

本仓库信息基于公开资料整理，国产 GPU 生态变化较快，具体细节请以各厂商官网公告为准。