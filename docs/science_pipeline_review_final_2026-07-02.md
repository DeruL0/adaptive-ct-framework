# 自适应 CT 重建最终研究审计（2026-07-02）

## 结论

当前可以成立的主方法不是“完整 hp-RD 框架”，而是一个更窄、更干净的
**全局跨层 jump-rate 自适应 h 重建**：

1. 用一个粗 packed octree 定义初始函数空间；
2. 所有叶节点存储非负 degree-0 Bernstein 系数；
3. 用唯一的投影域 fidelity 联合优化全部系数；
4. 在收敛平台触发细化；
5. L0、L1、L2 所有可细化叶节点使用同一个尺度归一化面跳变指标竞争；
6. 只由统一的新增字节预算截断全局排序；
7. 细化后继续全局联合优化，最后导出 float16 packed hierarchy。

正式方法不包含 support mask、材料阈值、最低分辨率、逐层配额、强制覆盖、
held-out topology gate、test-view 选择、TV、continuity、Sobolev loss、p-refinement
或 MACT。它们或者是被否决的技巧，或者当前没有消融证据支持。

Figure473 上，纯自适应 selector 在两个 uniform-WLS seed 上稳定；相对
balanced-weight 消融，移除该 heuristic 后 lockbox projection 和 full-volume
反而提升。它接近使用材料阈值和固定 split fraction 的强制覆盖基线；但
Fandisk 的第二对象实验显示，其区域值精度尚未泛化。因此当前结果足以支持
“无阈值、全局跨层自适应可行”，不足以支持“普遍优于已有方法”或“hp 自适应
有效”。

## 1. 不可违反的方法约束

正式配置是
`configs/research/figure473_128v_pure_adaptive.yaml`，它不继承其他配置。

禁止在目标运行中使用：

- `min_mu_threshold` 或任何材料/support 阈值；
- 从投影估计 support 后强制盒外为零；
- 强制所有材料单元至少达到某个 level；
- L0/L1/L2 的固定 active fraction 或分层 quota；
- 为某个 level 预留叶节点；
- 以 validation/test GT 或体数据决定拓扑；
- 只为改善视觉结果而增加的 smoothing、sharpening 或后处理；
- 根据当前对象手工指定空间区域。

允许的先验只有：

- 初始根网格，等价于有限维初始函数空间；
- 最大可用层数，等价于表示字典；
- 全局模型码率/叶节点预算；
- 训练收敛平台，决定何时比较动作，不决定细化哪里。

这一区分很重要。根网格不等于“最低有效分辨率”：未被选择的区域永久停留在
L0；被选择的区域可继续进入 L1/L2/L3。不存在任何区域被强迫达到后续层级。

## 2. 真实执行流水线

### 2.1 数据协议

- Figure473：128 train、16 validation、16 lockbox test；
- 三组角度互不重合，均定义在 `[0, π)`；
- detector 为 `1024 × 1024`；
- test views 不参与训练、细化或超参数选择；
- 合成体指标统一把预测与 GT 采样到 `256³` voxel centres；
- 主 seed 为 `20260702`，复现实验 seed 为 `20260703`。

### 2.2 表示与前向模型

根层为 `48³`，可用层级为 `[48, 96, 192, 384]`。每个叶节点保存一个非负
常数 Bernstein 系数。CUDA 从 packed hierarchy 逐层查询叶节点，不建立
`384³` 或更高分辨率的稠密索引。射线按真实叶边界分段，单元线积分在当前
degree 下精确计算。

正式训练目标只有 uniform projection WLS。无动态范围加权、TV、continuity
或 projection-gradient 辅助项。此前的 dynamic-range-balanced fidelity 含有
手工 `target_power/min/max/blend`，因此降为权重消融，不再属于无 trick 主线。

### 2.3 全局自适应判据

对任意可细化叶节点 \(K\)，定义

\[
\eta_K^2 = |K|\sum_{F\subset\partial K}[\mu]_F^2 ,
\]

其中 \([\mu]_F\) 是与相邻叶节点在共享面上的系数跳变，\(|K|\) 消除单元尺度
对原始 jump 的直接偏置。每次细化一个 degree-0 叶节点产生 8 个子节点，净增
7 个系数及其拓扑字节，因此排序使用同一 rate 口径。

每轮把 L0、L1、L2 的全部 eligible leaves 放入同一个队列。算法没有先
“覆盖 L0”，也没有再“分配 L1 比例”；预算只选择全局分数最高的动作。细化
使用 de Casteljau 初始化，动作瞬间保持函数不变，随后由同一个投影目标联合
优化全部系数。

当前配置使用三次全局新增原始字节预算：8 MB、25 MB、50 MB。它们是全局
rate budget，不是层级覆盖规则。

### 2.4 导出

最终导出 compact v3：

- packed `node_child_base:int32`；
- packed `node_leaf_id:int32`；
- 任意 dyadic 层数和各向异性 level shape 元数据；
- float16 coefficients。

训练 checkpoint 还保存优化器、训练历史和冗余 Python/Torch 状态，因此不应
与部署模型尺寸混为一谈。

## 3. 核心结果

| 方法 | Test PSNR | Test MAE | Full-volume PSNR | Material PSNR | Leaves | Compact |
|---|---:|---:|---:|---:|---:|---:|
| 强制覆盖基线 25k（含阈值，非正式） | 47.9325 | 0.001562 | 37.3836 | 30.4371 | 6.000M | 19.19 MB |
| **正式 uniform，seed 20260702** | **47.7482** | **0.001490** | **37.0289** | **30.0503** | 5.979M | 20.48 MB |
| **正式 uniform，seed 20260703** | **47.8027** | **0.001488** | **37.1621** | **30.1946** | 5.979M | 20.48 MB |
| balanced 权重消融，seed 20260702 | 47.2936 | 0.001499 | 36.7157 | 30.2081 | 5.979M | 20.51 MB |
| balanced 权重消融，seed 20260703 | 47.2096 | 0.001510 | 36.6519 | 30.1556 | 5.979M | 20.50 MB |
| Haar-GN h-only，48 root | 45.6739 | 0.001973 | 34.3400 | 26.7118 | 5.979M | 20.23 MB |
| 一次性贪心 unified hp | 43.8285 | 0.002132 | 33.6698 | 25.9328 | 0.111M | 1.55 MB |
| 各向异性 root 消融 | **48.6885** | 0.001456 | 37.1025 | 29.5617 | 6.000M | 19.11 MB |

两次正式 uniform 运行的差异为：

- test PSNR：0.055 dB；
- full-volume PSNR：0.133 dB；
- material PSNR：0.144 dB。

这说明正式方法在 Figure473 上对 seed 稳定。seed 20260702 的最终层级分布
不是预设的，得到：

- L0：28,707；
- L1：396,385；
- L2：1,571,757；
- L3：3,982,424。

三轮 uniform 动作的 parent-level 来源分别为 `[80808, 0, 0]`、
`[1003, 251522, 0]`、`[74, 7173, 497803]`。尤其第三轮仍同时选择
L0、L1、L2，证明执行路径不是伪装成自适应的固定 coarse-to-fine schedule。
训练后诊断显示：两个 seed 中按 GT material-mean 定义的根单元均有 0% 留在
L0，但按更宽的 GT support 定义仍分别有 12.83% 与 12.47% 留在 L0。这一分布
是 selector 自己产生的，不是 force-split。

## 4. 被消融否决的机制

### 4.1 support floor / 最低分辨率

材料阈值和 force-split 能改变空间覆盖，但其结论依赖当前对象的衰减值定义，
并把“自适应发现结构”替换为“先告诉算法哪里是物体”。即使某些指标更高，
也不能作为目标方法。相关结果只保留为被否决的上下文基线。

### 4.2 局部 Haar Gauss-Newton gain

该判据估计动作对当前 loss 的即时二阶下降，但在长时联合优化后排序失真。
48-root H-GN 最终 full-volume 只有 34.34 dB；其 64.6% 的真实材料根单元仍
停留在 L0，而强制覆盖基线为 0.87%。这不是通过改阈值可修复的问题：局部即时
gain 没有计算一次 h-split 为未来动作创造的 option value。

### 4.3 一次性贪心 hp

同一 added-byte 口径下，第一轮选择了 107,957 个 p 动作而只有 43 个 h 动作。
早期 p-elevation 获得较高即时 gain，但锁定了后续 h 路径，最终 volume PSNR
降至 33.67 dB。当前实验不能证明 p-refinement 有效，因此 p 不进入主方法。

### 4.4 各向异性根网格

各向异性配置把 test projection 提高到 48.69 dB，却把 material PSNR 降到
29.56 dB。这再次证明投影一致性和体积真实性会反向。实现本身保留，因为它
通过了 CPU/CUDA/compact 测试；实验结论不支持把它设为默认方法。

### 4.5 continuity、L3、权重 schedule

旧受控消融显示：

- p 相对 h-only 约降低 0.02 dB，并增加约 188k coefficients；
- L3 提升 projection 约 0.153 dB，却降低 full/material volume；
- continuity-off 略好且更快；
- uniform fidelity 比固定 balanced fidelity 差；
- 25k 明显优于 15k，旧“15k 已收敛”判断不成立。

因此这些模块不能因为理论上听起来合理就进入正式流程。

## 5. 第二对象检验

Fandisk 使用 48 train / 12 test、`96³` GT，纯全局 jump 在约 500k 参数下：

- test projection：39.1436 dB；
- material volume：16.0381 dB，MAE 0.08317；
- high-gradient MAE：0.09580；
- gradient ratio：1.392；
- compact：1.68 MB。

旧 dynamic context 使用不同目标，不能当严格受控对照；它的 material PSNR
为 17.2300 dB、MAE 0.04028，但 gradient ratio 为 2.271、high-gradient MAE
为 0.13977。纯 jump 的边缘梯度更接近 GT，但区域衰减值明显更差。

所以第二对象不是“通过”。它说明 jump estimator 偏向不连续性，对均匀区域的
绝对值恢复缺少同等强的风险估计。这一缺陷必须通过统一的目标导向误差估计器
解决，不能再添加 support floor 或材料专用规则。

## 6. 可声称与不可声称

当前可声称：

- 实现了不依赖稠密体、support mask 和最低分辨率的全局跨层自适应；
- packed CUDA hierarchy 支持任意 dyadic 深度及各向异性 shape；
- Figure473 上两个正式 uniform-WLS seed 的结果稳定；
- 全局 jump-rate selector 显著优于测试过的局部 H-GN 与贪心 hp selector；
- compact float16 模型约 20 MB。

当前不可声称：

- p-refinement 带来有效收益；
- hp 策略优于 h-only；
- detector Nyquist 自动证明 L3 必要；
- projection PSNR 提升等于体积重建更准确；
- 当前 selector 已跨对象泛化；
- 当前方法优于 NeAT 或其他文献方法，除非使用相同数据、视角、预算和指标重跑。

## 7. 下一步应解决的理论问题

下一步不是继续调 split fraction，而是替换动作价值定义：

1. 建立目标导向的后验风险或全局 surrogate，联合衡量界面 jump 与区域值误差；
2. 让 h、p、merge 成为可逆动作，显式计算未来 option value，而非一次性贪心；
3. 比较完整 rate-distortion curve，而不是只比较一个 6M-leaf 点；
4. 每个关键点至少 3 seeds，并在第二对象上使用完全相同协议；
5. 若引入 Poisson 统计模型，必须替换并统一训练目标与 action score，不能作为
   单独权重补丁。

这条研究路线仍保持一个目标、一个 estimator、一个全局 selector、一个预算，
而不是继续堆叠对象相关规则。

## 8. 复现与证据

- 正式配置：`configs/research/figure473_128v_pure_adaptive.yaml`
- 正式 uniform checkpoint：
  `output/research_figure473_pure_global_jump_uniform_48root_128v_25k/checkpoint.pt`
- 正式 uniform training report：
  `output/research_figure473_pure_global_jump_uniform_48root_128v_25k/training_report.json`
- balanced 权重消融：
  `output/research_figure473_pure_global_jump_48root_128v_25k`
- 第二 seed：
  `output/research_figure473_pure_global_jump_uniform_48root_128v_seed20260703_25k`
- Fandisk：
  `output/research_fandisk_pure_global_jump_48v_500k`
- 机器可读汇总：`output/research_audit/ablation_summary.json`

最终验收必须同时报告 projection、full volume、material volume、boundary
指标、模型字节数、运行时间和 seed；只报告 viewer 截图或 projection PSNR
不构成科学验证。
