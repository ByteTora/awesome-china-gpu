# awesome-china-gpu

国产 GPU/AI 芯片推理引擎与开发者生态指南。

持续追踪华为昇腾、寒武纪、沐曦、壁仞、摩尔线程等国产芯片在主流推理框架（vLLM、SGLang）上的支持情况，梳理落地实践中的"四堵墙"——生态墙、显存墙、通信墙、精度墙，并提供开发者上手实践路径。

---

## 国产 GPU 推理引擎对照表

| 企业 | 核心平台 | SGLang 支持 | vLLM 支持 | 自研推理引擎 |
|------|----------|-------------|-----------|-------------|
| **华为昇腾** | CANN | [官方原生支持](https://docs.sglang.io/hardware-platforms/ascend-npus/ascend_npu) | [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend) | MindIE |
| **寒武纪** | Cambricon NeuWare / BANG | ❌ 不支持 | [Cambricon/vllm-mlu](https://github.com/Cambricon/vllm-mlu) | MagicMind |
| **沐曦股份** | MetaX / MXMACA | ❌ 不支持 | [MetaX-MACA/vLLM-metax](https://github.com/MetaX-MACA/vLLM-metax) | — |
| **壁仞科技** | BIRENSUPA | ❌ 不支持 | ❌ 暂无公开仓库 | — |
| **摩尔线程** | MUSA | ❌ 不支持 | [MooreThreads/vllm-musa](https://github.com/MooreThreads/vllm-musa) | — |

> **说明：** 截至 2026 年 6 月，SGLang 框架官方原生硬件支持列表中，国产芯片仅包含华为昇腾（Ascend NPU）。其余厂商目前仅支持 vLLM 或其自研推理引擎。

---

## 厂商详情

### 华为昇腾（Huawei Ascend）

国产 AI 芯片领军者，全栈自研（硬件 + CANN 软件栈 + MindIE 推理引擎）。

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

**关键信息：**
- 免费算力：华为开发者空间提供 100 卡时/人或 180 小时免费资源
- SGLang 和 vLLM 均有官方支持，是国产卡中生态最完整的
- CANN 与 CUDA 差异较大，软件迁移有一定学习成本
- 硬件路线：910B → 910C（对标 H100）→ 950 系列（预计 2026 Q4 量产）

---

### 寒武纪（Cambricon）

以推理见长，思元系列在 vLLM 推理框架上表现突出。

| 资源 | 链接 |
|------|------|
| 开发者社区 | https://developer.cambricon.com/ |
| GitHub 组织 | https://github.com/Cambricon |
| vLLM-MLU | https://github.com/Cambricon/vllm-mlu |
| MagicMind 推理引擎 | https://developer.cambricon.com/sdk/magicmind |

**关键信息：**
- 思元 590 FP16 算力 345 TFLOPS，接近 A100 水平
- vLLM 实现支持 TP/PP/SP/DP/EP 5D 混合并行、PD 分离部署
- 性价比突出，成本约为昇腾 910B 的 60%
- 第三方算力平台：矩池云（MatPool）、AutoDL 有寒武纪实例
- 不推荐 SGLang——截至 2026 年 6 月无官方/社区可用分支

---

### 沐曦股份（MetaX）

GPGPU 路线，MXMACA 软件栈在 API 层面高度兼容 CUDA，降低迁移成本。

| 资源 | 链接 |
|------|------|
| 开发者社区 | https://developer.metax-tech.com/ |
| 模力方舟（ModelZoo） | https://developer.metax-tech.com/ai-models/ |
| vLLM-MetaX | https://github.com/MetaX-MACA/vLLM-metax |
| 文档中心 | https://developer.metax-tech.com/api/client/document/preview/1222/index.html |
| Gitee AI 平台 | https://ai.gitee.com/ |

**关键信息：**
- 已适配 6000+ CUDA 应用，覆盖 500+ AI 模型
- 对熟悉英伟达生态的开发者相对友好
- Gitee AI 提供曦云 C500 实例，可零成本部署模型
- 新注册用户/高校师生可领免费算力券

---

### 壁仞科技（Biren Technology）

壁砺系列 GPGPU，主打大算力训推一体。

| 资源 | 链接 |
|------|------|
| 开发者社区 | https://developer.birentech.com/ |
| BIRENSUPA 平台 | https://www.birentech.com/product/software/birensupa/ |

**关键信息：**
- 壁砺 166 系列可支持万亿参数模型训练与推理
- 已通过国家《安全可靠测评》I 级认证
- 下一代 BR20X 芯片计划 2026 年商用
- 生态较封闭，vLLM/SGLang 公开仓库暂未找到，需关注官方邀测

---

### 摩尔线程（Moore Threads）

全功能 GPU 路线，MUSA 架构，社区驱动为主。

| 资源 | 链接 |
|------|------|
| 开发者平台 | https://developer.mthreads.com/ |
| 文档中心 | https://docs.mthreads.com/ |
| vLLM-MUSA | https://github.com/MooreThreads/vllm-musa |
| vLLM-MUSA 文档 | https://docs.mthreads.com/vllm-musa/vllm-musa-doc-online/intro/ |
| AutoDL 摩尔线程专区 | https://www.autodl.com/ |

**关键信息：**
- AutoDL 上提供 30 天免费试用，门槛最低，适合立刻上手
- 通过 torchada 兼容层实现 CUDA → MUSA 迁移
- 路线更接近"全功能 GPU"，场景更广，但纯 AI 训练场景需全面评估

---

## 落地挑战：国产算力的"四堵墙"

> 参考自 [硅基流动：国产算力落地大模型推理的四大挑战](https://industry.caijing.com.cn/20260326/5149782.shtml)

将大模型推理从"纸面算力"落地到"业务价值"，中间隔着四堵墙：**跑不通、跑不快、跑不稳、跑不准**。

### 1. 生态墙 → 跑不通

CUDA 生态深度绑定，国产芯片软件栈需要代码转译或重写。

| 挑战 | 说明 |
|------|------|
| CUDA 代码强耦合 | PagedAttention、FlashAttention 等关键算子直接写死为 CUDA Kernel，国产芯片需重写 |
| 算子缺失 | 新模型结构（如 MoE 的 Grouped GEMM）的算子支持滞后，可能导致回退执行、严重延迟 |

### 2. 显存墙 → 跑不快

大模型推理本质是 Memory-Bound（受限于访存带宽），而非 Compute-Bound。

| 挑战 | 说明 |
|------|------|
| 带宽瓶颈 | 国产芯片 HBM 带宽通常低于 H100/A100（3TB/s+），解码阶段 GPU 在"等数据" |
| 容量瓶颈 | 早期国产单卡显存较小（32G/64G），大模型必须多卡切分，引入通信开销 |

### 3. 通信墙 → 跑不稳

多卡并行时，卡间互联速度决定推理服务生死。

| 挑战 | 说明 |
|------|------|
| 互联带宽差距 | NVLink/NVSwitch 达 900GB/s，国产芯片多依赖 PCIe Gen5（约 128GB/s）或私有协议 |
| 通信库成熟度 | NCCL 历经多年迭代，国产通信库（HCCL/CNCL）在高并发下仍有鲁棒性优化空间 |

### 4. 精度墙 → 跑不准

底层优化与精度支持的差距，是"能用"到"好用"的分水岭。

| 挑战 | 说明 |
|------|------|
| FlashAttention 适配 | 不同国产芯片 SRAM 架构差异大，生搬硬套算法性能可能不升反降 |
| BF16/FP8 支持 | 英伟达原生支持 BF16/FP8，部分国产芯片以 FP16 为主，存在精度溢出风险 |
| INT4 缺失 | W4A16 是行业趋势，多数国产芯片对 INT4 的硬件加速支持尚在完善中 |

**应对思路：**
- **场景匹配：** 延迟不敏感、模型迭代较慢的任务，国产算力已是极具性价比的选择
- **投资人才：** 能深入硬件底层做 Kernel 优化、玩转分布式通信的团队是真正的护城河
- **拥抱协作：** 协同芯片厂商、MaaS 平台（如硅基流动）共同解决适配问题

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
- 新的算力租赁平台
- 实战部署经验 / 踩坑指南
- 性能对比 benchmark 数据

---

## 免责声明

本仓库信息基于公开资料整理，国产 GPU 生态变化较快，具体细节请以各厂商官网公告为准。