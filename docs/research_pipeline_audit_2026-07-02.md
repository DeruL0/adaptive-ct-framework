# CT 自适应重建流程研究审核（2026-07-02）

## 结论先行

当前工程有一个可靠的技术内核：packed hierarchy、Bernstein 非负系数、函数保持的 h/p 变换、CUDA 精确分段积分和 compact v3 导出都通过了数值测试。但截至本次审核前，**论文方法叙事不等于真实正式配置**，而且多个新增模块没有获得严格消融支持。

本轮在新的 `128 train + 16 validation + 16 lockbox test` 协议下重新测试后，默认流程应暂时缩减为：

1. 非负 degree-0 Bernstein 叶单元；
2. 一个固定的 dynamic-range-balanced projection fidelity；
3. 两轮 h-only coarse-to-fine refinement；
4. 固定拓扑后继续全局优化；
5. float16 packed hierarchy 导出。

当前 p-refinement、L3、face continuity、Sobolev projection loss 都不应出现在默认方法中。它们可以保留为实验代码，但不能作为论文贡献宣称。

这条简化流程不是最终理论终点。它仍有两个未解决的结构问题：h-refinement 依赖 material threshold 和固定比例；立方层级 `[96,192,384]` 没有匹配原始体数据 `[473,1024,1024]` 的各向异性采样。后续真正有研究价值的修改应围绕统一的 h/p 局部 Gauss-Newton action score 和各向异性 dyadic hierarchy，而不是继续增加独立 loss 或候选筛选补丁。

## 1. 审核范围与可复现协议

审核沿真实代码路径进行：

`load_r2_dataset -> random_training_rays -> BernsteinOctree.integrate_ray_batch -> projection fidelity -> topology update -> compact export`

新增研究数据协议：

- train：128 个独立平行束角度，均匀覆盖 `[0,π)`；
- validation：16 个角度，相对训练网格偏移 `1/3` 个训练角间隔；
- lockbox test：16 个角度，相对训练网格偏移 `2/3` 个训练角间隔；
- 三组角度互不重合；
- detector：`1024×1024`；
- volume：`473×1024×1024`；
- 训练过程只报告 validation；test 只在训练结束后评估一次；
- 所有消融使用 seed `20260702`；
- 体积比较统一把模型与 GT 采样到 `256³` voxel centres。

机器可读结果位于 `output/research_audit/ablation_summary.json`。

## 2. 当前工程中值得保留的核心

### 2.1 表示与投影算子

- 每个叶节点保存非负 Bernstein 系数，softplus 参数化保证衰减非负；
- CUDA 与 Torch reference 都按真实叶单元边界分段；
- 单元内使用足阶 Gauss-Legendre 积分，对当前低阶多项式是精确的；
- h-split 通过三维 de Casteljau 初始化，p-elevation 通过 Bernstein degree-elevation identity 初始化，动作瞬间保持函数不变；
- packed hierarchy 不创建 `768³` 稠密索引；
- compact v3 支持任意深度和 mixed degree，旧 v1/v2 仍可读。

这些是方法最坚实的部分，也与 NeAT/ACORN 的自适应层级表示方向一致，但本项目的区别应放在“局部多项式有限元式表示 + 可解释 action selection”，而不是泛称 octree。NeAT 用自适应显式结构与 neural features 集中复杂区域采样；ACORN 用 multiscale block decomposition 分配表示容量。本项目不需要复制它们的 decoder，只需要证明自己的局部 action estimator 更有效。

### 2.2 工程测试

本轮完整测试为 `68/68` 通过，覆盖：

- CPU/CUDA forward、backward、segment contribution；
- 四层 packed topology；
- h/p 函数保持；
- 2:1 balance；
- compact v1/v2/v3；
- viewer slice 与真实投影路径；
- refinement score 与投影协议。

测试通过只证明实现局部不变量正确，不证明当前 refinement policy 能恢复真实体积；二者必须区分。

## 3. 研究流程中的结构问题

### 3.1 正式配置是 12 层继承链

审核前的正式配置从 `figure473_dicom_orig_accuracy_128unique_force_material_haar.yaml` 向上继承 12 层。最终行为包含大量被覆盖或 `enabled:false` 的历史 milestone。这样的配置无法让读者判断某个结果究竟由哪个机制产生。

本轮新增：

- `configs/research/figure473_128v_reference.yaml`：无继承的旧流程锁定版；
- `configs/research/figure473_128v_simple.yaml`：无继承的简化主线；
- `configs/research/ablations/*.yaml`：只继承一层且每个文件只改变一个因素。

### 3.2 README、pipeline 文档与正式运行不一致

README 描述的是 80/20 held-out action gate、merge、degree reduction、MACT 和 surface uncertainty；审核前正式配置实际关闭通用 adaptive engine、MACT 和 surface，使用 milestone-based Haar h-score 与 gradient p-score。`docs/current_algorithm_pipeline.md` 仍描述 16-view 旧实验，而且存在编码损坏。

论文和 README 必须只描述 canonical 配置真实执行的路径。实验性模块放在“exploratory alternatives”，不能与主方法混写。

### 3.3 两套 adaptation engine 并存

`train.py` 同时包含：

- `adaptive_projection_round`：held-out、forward/reverse action、merge/reduce；
- milestone engine：coefficient/material split、residual/Haar split、gradient p-elevation。

正式配置只使用后者，但 README 主要描述前者。`train.py` 已超过 2100 行，候选生成、评分、执行、训练和报告耦合在一起。后续应把 action space、estimator、budget selector、executor 拆为四个明确模块，只保留一套主引擎。

### 3.4 test leakage 已发生

旧训练每 1000 步评估 4 个 test views，且多轮配置选择直接比较 test PSNR。这些视角已经承担了 development set 的角色，不能继续称为最终测试。

本轮新增 `proj_val` 数据接口和 lockbox test。训练报告明确记录 `test_used_during_training`；所有新消融均为 `false`。

### 3.5 “15000 步已收敛”不成立

旧正式模型从 10k 到 15k 的 projection loss 仍下降约 25%。简化模型从 15k 训练到 25k 后：

- test projection：`47.1166 -> 47.9325 dB`；
- full volume：`36.8605 -> 37.3836 dB`；
- material volume：`29.8082 -> 30.4371 dB`；
- 25k 最后 1k validation 仍提升约 `0.047 dB`。

因此后续应使用 validation plateau + 最大迭代上限，或至少报告 convergence curve，不能用固定步数直接命名“收敛版本”。

### 3.6 投影指标与真实体积指标会反向

L3 消融是最直接证据：

- L3 令 lockbox projection `+0.153 dB`；
- full-volume 却 `-0.056 dB`；
- material-volume `-0.088 dB`；
- 增加约 1.33M leaves。

这说明更多自由度可以改善可见线积分，却沿投影零空间偏离真实体积。最终方法必须把 full/material volume 作为合成数据上的主要验证指标，projection PSNR 只能作为 data consistency 指标。

## 4. 消融结果

| Variant | Test proj. PSNR | Full volume PSNR | Material PSNR | Leaves | Compact | Time |
|---|---:|---:|---:|---:|---:|---:|
| reference: hp + L3 + continuity + schedule | 47.1772 | 36.7417 | 29.6397 | 7.328M | 24.36 MB | 255 s |
| h-only + L3 + continuity + schedule | 47.1974 | 36.7428 | 29.6395 | 7.328M | 24.06 MB | 234 s |
| hp + no-L3 + continuity + schedule | 47.0241 | 36.7979 | 29.7275 | 6.000M | 19.42 MB | 232 s |
| h-only + no-L3 + continuity + schedule | 47.0417 | 36.7991 | 29.7274 | 6.000M | 19.10 MB | 208 s |
| h-only + no-L3 + no-continuity + schedule | 47.0426 | 36.8006 | 29.7292 | 6.000M | 19.10 MB | 168 s |
| h-only + no-L3 + uniform | 46.5922 | 36.7972 | 29.6542 | 6.000M | 19.09 MB | 211 s |
| simple static balanced, 15k | 47.1166 | 36.8605 | 29.8082 | 6.000M | 19.11 MB | 170 s |
| simple static balanced, 25k | **47.9325** | **37.3836** | **30.4371** | 6.000M | 19.19 MB | 322 s |

### 4.1 p-refinement

当前 p 策略令 test 与 validation 都降低约 `0.020 dB`，full volume 也无收益，同时增加约 188k coefficients 和 21 s。根因不是“p 一定无效”，而是当前策略并非真正统一的 R-D 决策：

- p 先用 coarse residual quantile band 筛候选；
- 再用新函数子空间的 gradient energy 排序；
- 没有 Hessian/Fisher normalization；
- 没有绝对 rate threshold；
- 没有让同一单元的 h 与 p 直接竞争。

因此当前实现只能称为 gradient-ranked p allocation，不能称为完整 R-D hp optimization。

### 4.2 L3

当前 L3 提升投影一致性但降低体积准确度，应默认关闭。不能据此断言 L3 永远无效；正确结论是当前 Haar-diagonal score 没有识别“对真实体积有用”的 L3 自由度。

### 4.3 continuity

关闭 continuity 后所有指标略升，时间从 208 s 降到 168 s。当前实现固定缓存最多 8192 个 face pairs，相对数百万界面覆盖极低且不是每步无偏重采样。它既不能代表全局连续性，也不值得 19% 的运行开销，应删除出默认方法。

### 4.4 fidelity

uniform fidelity 的 full-volume 与 balanced 接近，但 projection 和 material 指标更差。静态 balanced 又优于两阶段 schedule，因此保留一个固定公式即可：

\[
w(y)=\operatorname{normalize}\left[(1-\alpha)(|y|+\epsilon)^{-1}+\alpha(|y|+\epsilon)^\gamma\right]
\]

并进行上下界截断。它应称为 dynamic-range-balanced fidelity，而不是 Poisson WLS。只有在输入包含 photon counts 和噪声模型时，才应使用真正由似然推导的统计权重。

## 5. 与 NeAT 的本地实测

官方 NeAT 在同一 128-view 数据上完成真实训练。由于 projection test protocol 不同，新 lockbox projection 不直接与旧 NeAT 4-view 数字比较；体积可在同一 `256³` 协议比较：

| Method | Full volume PSNR | Material PSNR | Core/compact size |
|---|---:|---:|---:|
| NeAT | 34.8789 | 27.0329 | 40.71 MB |
| 本项目 simple 25k | **37.3836** | **30.4371** | **19.19 MB** |

本项目在体积上约高 `2.50 dB`，材料区约高 `3.40 dB`，文件约为 NeAT core 的 47%。这说明显式叶单元方向有价值。但 NeAT 的旧 4-view projection PSNR 更高，恰好再次说明 projection fit 不能代替 volume fidelity。

## 6. 建议的统一理论流水线

### 6.1 单一目标

对固定表示空间 \(V(\mathcal T,p)\)，只优化一个数据项：

\[
\min_{c\ge 0}\; \frac{1}{N}\sum_r w(y_r)\,[A_{\mathcal T,p}c-y]_r^2.
\]

不再叠加 Sobolev、固定 face continuity、material force loss 等相互独立的修补项。若真实噪声实验需要先验，只允许增加一个有明确统计解释的 hierarchical detail prior，并单独消融。

### 6.2 统一 action space

每个 degree-0 leaf 同时有三个候选：

- keep；
- h-split：8 个 child constants，新增 7 个 Haar detail DOFs；
- p-elevate：`[1,1,1]`，新增 7 个 trilinear complement DOFs。

两种动作从同一残差、同一 rays、同一局部 Gauss-Newton 模型计算：

\[
G_a(C)=\frac{1}{2}g_a^\top(H_a+\epsilon I)^{-1}g_a,
\qquad
S_a(C)=\frac{G_a(C)}{\Delta R_a}.
\]

其中 \(\Delta R_a\) 使用 compact float16 实际新增字节。每个单元只执行 score 更高的动作，全局在字节预算下选择。这样不再需要：

- `residual_quantile_range`；
- `active_fraction` 分别调 h 与 p；
- `min_mu_threshold` force-split；
- p-only gradient energy；
- h-only Haar score；
- 事后解释“这里应该 h、那里应该 p”。

这与 goal-oriented hp-adaptivity 的基本思想一致：用目标函数相关的局部误差估计选择新增空间，而不是用单纯 gradient magnitude 或密度阈值。自适应 wavelet inverse methods 也强调用适应未知解的较小空间降低计算量，而不是无约束增加高频自由度。

### 6.3 各向异性层级

原始 volume 为 `[473,1024,1024]`，当前 cubic finest level 384 或 768 在两个 1024 轴上欠分辨，同时在 473 轴上浪费 L3 容量。建议把 root/level shape 改成：

`[60,128,128] -> [120,256,256] -> [240,512,512] -> [480,1024,1024]`

它保持每次三轴二分和 packed hierarchy 结构不变，但最高层直接匹配原始采样。root cells 约 0.98M，与当前 `96³≈0.885M` 同量级。这比盲目增加 cubic L3 更符合采样几何。

### 6.4 收敛与停止

- topology update 只能在 validation objective plateau 后发生；
- 当前第一轮仍在 `max_iteration=3000` 强制触发，当时相对改善 2.4%，不是 plateau；
- 每次动作后重建 optimizer；
- 最后一轮 topology 后用 validation plateau 停止，test 不参与；
- 报告完整 rate-distortion-convergence curve。

## 7. 下一阶段必须跑的实验

1. **各向异性层级 vs cubic 层级**：同 root-cell 数、同 compact-byte budget；
2. **统一 estimator 的 h-only vs hp**：同一 action pool、同一 byte budget；
3. **diagonal vs full `7×7` local Hessian**：验证对 action ranking 的影响；
4. **validation score 与实际 short-tune gain 的相关性**：Spearman/Pearson 和 top-k precision；
5. **25k/35k/plateau**：确定真正停止点；
6. **至少 3 seeds**：报告均值与标准差；
7. **第二个对象**：避免 Figure473 单对象过拟合研究结论；
8. **noise study**：无噪声、Poisson noise 两档，分别使用 balanced 与真实 likelihood 权重。

## 8. 代码整理顺序

1. 把旧配置移入 `configs/legacy/`，研究主配置只保留 `research/`；
2. 拆分 `train.py`：trainer、action estimator、budget selector、executor、reporter；
3. 将 `projection_domain.py` 中未启用的旧 adaptive engine 移入 `experimental/`；
4. 实现 anisotropic level shapes；
5. 实现统一 h/p local Hessian estimator；
6. 加 validation plateau 与多 seed runner；
7. 重写 README 和 pipeline 文档，使其只描述 canonical 路径；
8. 建立首个 Git baseline commit；当前仓库没有任何 commit，全部文件处于 untracked 状态，不满足可复现研究工程要求。

## 参考方向

- NeAT: Neural Adaptive Tomography — https://arxiv.org/abs/2202.02171
- ACORN: Adaptive Coordinate Networks — https://arxiv.org/abs/2105.02788
- Adaptive differentiable grids for cryo-electron tomography — https://academic.oup.com/bioinformaticsadvances/article/3/1/vbad131/7280741
- Adaptive Wavelet Galerkin Methods for Linear Inverse Problems — https://epubs.siam.org/doi/10.1137/S0036142902411793
- Inverse radiative transfer with goal-oriented hp-adaptive mesh refinement — https://doi.org/10.1088/1361-6420/acf785
