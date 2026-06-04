# awesome-china-gpu

追踪国产 GPU/AI 芯片在 LLM 推理（vLLM、SGLang）上的适配进展与开源仓库。

当企业将大模型推理从英伟达迁移到国产芯片时，普遍面临**跑不通、跑不快、跑不稳、跑不准**四大挑战。本仓库持续收录华为昇腾、寒武纪、沐曦、壁仞、摩尔线程等厂商在主流推理框架上的支持情况、官方仓库、开源项目与开发者上手资源，帮助算法工程师和技术决策者快速评估国产算力的落地可行性。

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

> **说明：** 截至 2026 年 6 月，SGLang 框架官方原生硬件支持列表中，国产芯片仅包含华为昇腾（Ascend NPU）。其余厂商目前仅支持 vLLM 或其自研推理引擎。本表将持续更新。

| 企业 | 核心平台 | SGLang 支持 | vLLM 支持 | 自研推理引擎 |
|------|----------|-------------|-----------|-------------|
| **华为昇腾** | CANN | [官方原生支持](https://docs.sglang.io/hardware-platforms/ascend-npus/ascend_npu) | [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend) | MindIE |
| **寒武纪** | Cambricon NeuWare / BANG | ❌ 不支持 | [Cambricon/vllm-mlu](https://github.com/Cambricon/vllm-mlu) | MagicMind |
| **沐曦股份** | MetaX / MXMACA | ❌ 不支持 | [MetaX-MACA/vLLM-metax](https://github.com/MetaX-MACA/vLLM-metax) | — |
| **壁仞科技** | BIRENSUPA | ❌ 不支持 | ❌ 暂无公开仓库 | — |
| **摩尔线程** | MUSA | ❌ 不支持 | [MooreThreads/vllm-musa](https://github.com/MooreThreads/vllm-musa) | — |

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

## 上手实践路径

如果你熟悉英伟达生态（vLLM / SGLang / CUDA），建议按"先易后难、先免费后付费"的原则循序渐进：

### 第一步：快速启动（零成本）

| 平台 | 厂商 | 免费资源 | 适合场景 |
|------|------|----------|----------|
| AutoDL | 摩尔线程 | 30 天免费试用 | 环境搭建、基础推理 |
| Gitee AI | 沐曦 | 免费额度 | 模型部署验证 |
| 华为开发者空间 | 华为昇腾 | 100 卡时/人 | 模型迁移、性能调优 |

### 第二步：深度探索

| 平台 | 厂商 | 说明 |
|------|------|------|
| 矩池云 / AutoDL | 寒武纪 | 付费实例，跑 vLLM-MLU 推理 |
| ModelArts | 华为昇腾 | 企业级训练调优 |

### 第三步：内容输出

围绕真实体验，输出以下类型内容：
- 环境搭建踩坑记录
- 模型迁移实战（CUDA → CANN / CUDA → MUSA）
- 性能对比（同模型在不同国产卡上的推理吞吐）
- 成本分析（国产卡 vs 英伟达推理成本对比）

---

## 商业模式参考

如果你是大模型算法工程师，想转型国产算力服务商，可以分三步走：

| 阶段 | 时间 | 核心动作 |
|------|------|----------|
| 快速启动 | 1-3 个月 | 输出技术博客/公众号建立权威；申请二级/三级代理商资质 |
| 验证模式 | 3-12 个月 | 推出标准化服务包（评估 + 部署 + 运维）；完成首批付费客户 |
| 构建壁垒 | 1-3 年 | 沉淀工程化工具链；建立异构算力调优知识库 |

**核心壁垒：** 算法工程师对模型的理解 + 对国产异构算力的深入实践，而不是硬件或资金。

---

## 生态协同

上海 AI 实验室的 DeepLink 混合推理加速方案已实现对华为昇腾、阿里平头哥、沐曦、壁仞等多款国产 GPU 的混合调度。多模态生成、高并发服务等场景中，比单一芯片方案推理时延最大可优化 34.5%，推理吞吐量最大可提升 32%。

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