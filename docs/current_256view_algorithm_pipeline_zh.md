# 当前 256-view 自适应 CT 算法流水线

> 核验日期：2026-07-07  
> 对应配置：`configs/research/industrial_detail_v5_256v_equal_ratio_anisotropic.yaml`  
> 对应结果：`output/industrial_detail_v5_256v_equal_ratio_anisotropic/checkpoint.pt`

## 1. 结论与版本边界

当前 Viewer 中视觉效果最好的 256-view 结果，实际运行的是：

**各向异性 packed Bernstein octree + p0 叶函数 + CUDA 精确射线积分 + uniform 投影 MSE + 收敛触发的全局 jump/byte h-refinement。**

当前结果没有启用：

- nodal/shared-corner 参数化；
- p1 三线性叶函数；
- Ambrosio–Tortorelli 相场；
- split/merge/untie 循环；
- TV、continuity、Sobolev 或体数据监督；
- validation/test view 拓扑选择。

代码仓库包含上述实验模块，但它们不属于当前 256-view checkpoint 的生成路径。

`model.max_degree: [1,1,1]` 只是代码允许的最高次数，不表示实际叶节点已经是 p1。checkpoint 核验结果为：

```text
leaf_count        = 5,708,529
coefficient_count = 5,708,529
unique degrees    = [0,0,0] only
```

因此当前每个叶节点只存一个非负衰减系数。

## 2. 输入与数据协议

### 2.1 算法输入

- 256 张训练投影；
- 每张投影为 `1024 × 1024`；
- 角度在 `[0, π)` 内均匀采样，都是独立平行束方向；
- 扫描几何与探测器坐标；
- 4 张独立 test projection。

训练代码只加载投影数据，不加载 GT 体：

```text
training views   = 256
validation views = 0
test views       = 4
test in training = false
```

GT 体只在训练结束后的离线切片评价和 Viewer 对照中使用，不进入损失、细化或参数优化。

### 2.2 坐标域

物体包围盒归一化为：

\[
\Omega=[-1,1]^3.
\]

当前各向异性层级为：

| Level | 网格尺寸 |
|---|---:|
| L0 | \(22\times48\times48\) |
| L1 | \(44\times96\times96\) |
| L2 | \(88\times192\times192\) |
| L3 | \(176\times384\times384\) |

各层严格按三轴二分，因此仍然是完整的八叉分裂；“各向异性”只表示根网格三个轴的初始尺寸不同。

它与原始体 `473×1024×1024` 的轴向压缩倍率约为：

\[
473/176=2.688,\qquad 1024/384=2.667.
\]

## 3. 场表示

### 3.1 Packed hierarchy

拓扑不建立 `176×384×384` 稠密索引，而是使用 packed hierarchy：

- `node_child_base:int32`：内部节点指向连续八个子节点，叶节点为 `-1`；
- `node_leaf_id:int32`：节点对应的叶编号；
- `leaf_levels`：每个叶的层级；
- `leaf_coords`：每个叶在该层级中的整数坐标；
- `leaf_degrees`：每个叶的 Bernstein 次数；
- `coefficient_offsets`：叶系数在 packed coefficient vector 中的偏移。

当前训练全部为 p0，所以每个叶 \(K\) 表示：

\[
\mu_K(\mathbf x)=c_K,\qquad c_K\ge 0.
\]

### 3.2 非负性

优化变量不是直接的 \(c_K\)，而是 logit \(\theta_K\)。物理系数通过 softplus 得到：

\[
c_K=\operatorname{softplus}(\theta_K+s),
\]

其中配置中的 attenuation shift 为 \(s=-3\)。

因此非负性由参数化严格保证。当前代码使用普通 Adam，不是每步 clamp 的 projected Adam。

### 3.3 当前连续性性质

当前叶节点各自持有独立 p0 系数：

- 叶内部为常数；
- 叶与叶之间允许跳变；
- 没有共享顶点；
- 没有跨叶 continuity 约束。

这使材料界面不会被连续性正则强制抹平，但平滑区域仍可能出现分块台阶。

## 4. 初始化

初始拓扑是完整 L0 根网格，共：

\[
22\times48\times48=50,688
\]

个 p0 叶节点。

初始化使用训练投影的全局均值。归一化盒子的代表路径长度取 2，因此：

\[
c_K^{(0)}=\frac{1}{2}\operatorname{mean}(y_{\mathrm{train}})
\]

并对所有 L0 叶使用同一个初值。

这里没有 FDK、support mask、材料阈值或最低材料分辨率。

## 5. 精确投影前向

对射线 \(r\)，预测线积分为：

\[
\hat y_r
=\int_r\mu(\mathbf x)\,d\mathbf x
=\sum_{K\cap r\ne\varnothing}
\int_{r\cap K}\mu_K(\mathbf x)\,d\mathbf x.
\]

当前 p0 情况下：

\[
\hat y_r=\sum_K l_{r,K}c_K,
\]

其中 \(l_{r,K}\) 是射线在叶节点 \(K\) 内的真实线段长度。

训练时满足条件后自动走 native CUDA packed traversal：

- 输入 packed topology，而非稠密体；
- 按真实叶边界分段；
- forward/backward 都通过 native Bernstein CUDA 路径；
- `integration_mode=exact`；
- `samples_per_ray=1024` 仍保留在配置和 fallback 接口中，但 native exact 路径不依赖固定步长采样近似。

代码也支持 p1/p2/p3 的 Gauss 精确积分，但本 checkpoint 只使用 p0。

## 6. 唯一训练目标

每一步从 256 个训练视角的所有 detector pixels 中随机抽取：

\[
N_{\mathrm{ray}}=65,536
\]

条射线。

投影权重 schedule 只有 uniform 阶段，因此：

\[
w_r=1.
\]

实际损失是 batch mean squared error：

\[
\mathcal L_{\mathrm{proj}}
=\frac{1}{N_{\mathrm{ray}}}
\sum_r(\hat y_r-y_r)^2.
\]

当前所有附加项都为零：

\[
\mathcal L=\mathcal L_{\mathrm{proj}},
\]

具体包括：

- `tv_weight = 0`；
- projection-gradient loss = 0；
- volume loss = 0；
- coefficient continuity = 0；
- AT phase energy 未接入。

优化器为 Adam：

```text
iterations = 25,000
learning rate = 0.015
batch rays = 65,536
```

## 7. 收敛触发的 h-refinement

### 7.1 Plateau 检测

训练维护随机 ray batch projection loss 的 EMA：

- EMA span：50；
- 判定窗口：200；
- 相对改善阈值：0.2%；
- 最大允许相对回退：0.5%。

每一轮 refinement 同时受：

- 最早触发 iteration；
- 上一轮后的最小间隔；
- 最晚强制触发 iteration

约束。

当前代码的 plateau 尺子仍来自随机训练 batch 的 EMA，并不是后来实验模块中的 fixed decision-ray evaluator。

### 7.2 全局界面 jump 指标

对所有仍可分裂的叶节点，计算六个面的系数 jump，并使用：

\[
I_K
=|K|\sum_{F\subset\partial K}[\mu]^2_F.
\]

其中：

- \(|K|\) 是叶节点体积；
- \([\mu]_F\) 是该面两侧表示的失配；
- p0 情况下就是相邻叶衰减系数差。

体积因子使不同层级的候选处于同一尺度。

### 7.3 真实 raw-byte 近似

一次八叉 split 将一个父叶替换为八个子叶，净增加七个叶。选择器按 compact-v3 的 raw packed 布局估算：

\[
\Delta R_K=85+14n_K,
\]

其中 \(n_K\) 是父叶系数数目。当前 p0 的 \(n_K=1\)，因此：

\[
\Delta R_K=99\ \text{bytes}.
\]

动作率失真分数为：

\[
S_K=\frac{I_K}{\Delta R_K}.
\]

所有可分裂层级共同进入一个全局队列，按 \(S_K\) 降序选择，直到：

- 本轮新增字节预算耗尽；或
- 总叶节点上限 6,000,000 即将达到。

当前没有：

- active fraction；
- 分层 quota；
- support/material mask；
- 强制覆盖；
- held-out topology gate；
- test-view selector。

### 7.4 三轮实际细化

| 轮次 | 触发 iteration | 新 split 父节点 | 主要来源 | 预计新增 raw bytes |
|---|---:|---:|---|---:|
| 1 | 1,912 | 50,688 | L0 | 5,018,112 |
| 2 | 3,154 | 252,525 | L1 | 24,999,975 |
| 3 | 6,007 | 505,050 | L1/L2 | 49,999,950 |

split 使用函数保持初始化：八个子节点初始复现父节点当前函数，因此拓扑变化瞬间不会主动改变投影；随后继续用投影 MSE 联合训练。

每次拓扑变化后重新建立 Adam optimizer，旧的动量状态不会跨拓扑继承。

当前没有 merge/coarsening，因此早期 split 之后不能回收。

## 8. 最终模型

25,000 步结束后的叶节点分布：

| 层级 | 叶节点 | 比例 |
|---|---:|---:|
| L0 | 0 | 0% |
| L1 | 152,084 | 2.66% |
| L2 | 1,523,205 | 26.68% |
| L3 | 4,033,240 | 70.66% |
| 总计 | 5,708,529 | 100% |

对应：

```text
learnable coefficients = 5,708,529
checkpoint size         = 413.76 MiB
reported model bytes    = 485,982,548 bytes
peak CUDA allocated     = 2.92 GiB
peak CUDA reserved      = 3.16 GiB
training elapsed        = 303.37 s
```

checkpoint 较大是因为它保存训练友好的 int64 拓扑、offset、owner 等冗余张量；它不是部署码率。

## 9. 评价协议与当前结果

4 个 test views 只在训练完成后评价：

```text
projection PSNR = 42.7531 dB
projection MAE  = 0.0019240
```

另取 x/y/z 三轴共 15 张代表切片进行离线体域评价：

```text
slice PSNR             = 30.7708 dB
slice MAE              = 0.0122507
high-gradient PSNR     = 18.6225 dB
high-gradient MAE      = 0.0583883
gradient ratio         = 0.4270
distance to ratio 1    = 0.5730
```

这些体域指标使用 GT，只做评价，不反馈给算法。

## 10. Compact 导出

代码中的 compact packed-v3 导出会保存：

- packed hierarchy；
- level shapes；
- degree tags；
- narrow integer topology；
- float16 或其他配置量化后的物理系数；
- 少量 metadata。

它不保存：

- Adam 状态；
- autograd-friendly 冗余索引；
- 训练历史；
- 投影图像；
- GT 体。

当前配置继承结果要求 `float16 + topology:auto`，但本次 256-view 运行目录的 `exports` 为空，尚未生成 compact 文件。因此当前只能确认 checkpoint 大小，不能把约 20 MB 当作本次 256-view 已经实际导出的文件大小。

## 11. Viewer 的真实求值路径

### Slice

Slice 在原始体素中心构造三维坐标，分块调用：

```text
model.forward_mu(points)
```

因此它直接查询八叉树叶函数。当前模型为 p0，所以视觉上仍是分块常数；如果加载真实 p1 checkpoint，同一路径会计算 Bernstein 三线性函数，而不是显示叶均值。

### Stored train/test Projection

Viewer 的 train/test projection 页面调用 `render_split`，进一步调用模型的 `integrate_ray_batch`。CUDA 可用时走 native exact packed projector。因此这里显示的是模型真实射线积分。

### Arbitrary-angle Volume Projection

Viewer 中另一个“任意角度 volume projection”接口目前不同：

1. 先把模型解码成原始尺寸稠密体；
2. 再调用 dense parallel projector。

它不是 packed octree 的直接射线积分。论文或正式指标应使用 stored train/test projection 路径。

## 12. 当前算法的准确定位

当前可验证的方法应表述为：

> 从多视角工业 CT 投影直接优化的、受部署字节预算驱动的全局自适应 packed octree。算法以统一的尺度归一化界面 jump/byte 指标，在所有可细分层级间分配 p0 空间分辨率，并通过精确 CUDA 射线积分联合优化非负衰减场。

它当前实现了：

- 投影域自监督重建；
- 原始分辨率数据；
- 任意深度 packed CUDA traversal；
- 三轴各向异性 dyadic hierarchy；
- 无对象相关 mask/quota 的全局 h 分配；
- 明确的 raw-byte 动作成本；
- checkpoint/compact 分离；
- Viewer 的函数真实切片和 stored-view 投影。

它当前尚未实现于这份最佳 checkpoint：

- 平滑区域的 p1/nodal 函数压缩；
- merge 后悔机制；
- 相场驱动 split/merge/untie；
- fixed-ray 稳定 plateau/判决；
- 最终 compact 256-view artifact。

因此，当前视觉改善主要来自独立视角从 96 增加到 256，而不是 p1、nodal 或 AT 相场。
