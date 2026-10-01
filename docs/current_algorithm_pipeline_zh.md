# 工业稀疏视角 CT：界面感知共享顶点 p1 自适应函数场

本文档描述仓库当前实际实现并完成 96-view 验证的算法。失败消融不属于正式流水线，单独列在文末。

## 1. 算法目标

算法联合考虑三个目标：

1. 从有限角度工业 CT 投影中重建衰减场；
2. 减少常数体素造成的块状感，改善材料内部的平滑变化；
3. 在压缩模型中保留裂纹、孔隙、薄壁和多材料边界。

正式优化只使用扫描投影与几何信息。GT 体数据只用于最终评价，不参与训练、拓扑选择或界面选择。

当前验证数据为：

- 96 个独立平行束训练角度，均匀覆盖 `[0, π)`；
- 4 个与训练角度分离的测试角度；
- 探测器分辨率 `1024 × 1024`；
- GT 体分辨率 `473 × 1024 × 1024`，仅用于评价。

## 2. 总体流水线

```text
96-view 投影
    │
    ▼
h-only 自适应八叉树重建
    │  产生细节优先参考拓扑与衰减场
    ▼
p5 keep / p0 / p1 投影域树压缩与联合精调
    │  作为 nodal 方法的拓扑和函数初始化
    ▼
2:1 balance
    │
    ▼
全共享顶点 p1 场 + hanging-node 约束
    │
    ▼
投影域 private-leaf 动作评分
    │  在统一 raw-byte 预算下选择界面断开区域
    ▼
界面感知共享顶点 p1 场
    │
    ▼
固定拓扑、固定约束联合精调
    │
    ▼
Compact nodal v4 + Viewer 精确函数渲染
```

当前 nodal 阶段没有重新决定 h 拓扑，而是复用已经验证的 p5 拓扑。最终部署表示中所有叶都是 p1；p0/p1 区分只存在于初始化阶段。

## 3. 第一阶段：细节优先的自适应拓扑

基础空间为 packed octree：

```text
root = 48³
levels = [48, 96, 192, 384]
```

从 p0 常数叶开始，仅使用投影 WLS：

\[
L(\mu)=\frac12\lVert A\mu-y\rVert_2^2.
\]

其中：

- `A` 为真实平行束 X-ray 积分算子；
- `y` 为训练投影；
- 使用 uniform 权重；
- 系数通过 softplus 保持非负；
- 不使用 GT、support mask、TV、continuity 或边缘后处理。

训练达到 plateau 后，所有可分裂层级进入统一的 h-refinement 队列。该阶段优先建立足够细的参考拓扑，供后续函数化表示使用。

## 4. p5 初始化场

p5 对 h-only 参考树中的每个内部子树构造三个候选：

- `keep`：保留完整子树；
- `p0`：合并为一个常数函数；
- `p1`：合并为一个局部三线性 Bernstein 函数。

p1 叶内函数为：

\[
\mu_K(u,v,w)=
\sum_{i,j,k\in\{0,1\}}
c_{ijk}B_i(u)B_j(v)B_k(w).
\]

候选通过训练投影残差评分，再由树 DP 选择。确定拓扑和 degree 后，只优化系数。该模型不是最终 nodal 表示，而是为 nodal 方法提供：

- 已恢复细节的空间拓扑；
- 每个叶的一侧函数值；
- 材料界面附近的独立参考值。

当前初始化 checkpoint 为：

```text
output/industrial_detail_v5_96v_20mb/final_score32k/checkpoint.pt
```

## 5. 最终表示：界面感知共享顶点 p1 场

### 5.1 所有叶统一使用 p1

最终模型不再选择 p0 或 p1。每个叶都使用同一种三线性函数。均匀区域通过八个角点值相等自然表达，不需要单独的 p0 建模分支。

### 5.2 Master vertices

平滑相邻叶的同一几何角点共享一个 master attenuation value。规则网格内部，最多八个叶共享一个顶点，因此独立参数数目显著低于每叶保存八个系数。

叶局部角点系数由 master values 线性重建：

\[
c_{K,q}=\sum_m W_{K,q,m}v_m,
\qquad
\sum_m W_{K,q,m}=1,
\quad W_{K,q,m}\ge0.
\]

其中 `v_m` 是独立 master vertex，`W` 是共享关系或 hanging-node 约束。

### 5.3 2:1 balance 与 hanging nodes

八叉树先执行 2:1 balance，使相邻叶的层级差不超过一级。粗细交界处：

- 粗边中点由两个粗端点各以 `0.5` 权重插值；
- 粗面中心由四个粗角点各以 `0.25` 权重插值；
- hanging vertex 不作为独立参数。

因此所有被绑定的粗细交界严格满足 C0，不依赖 continuity loss。

### 5.4 显式界面断开

工业 CT 的空气—材料、多材料、裂纹和孔隙边界是真实不连续场。模型允许选中的叶保留八个 private vertices，不与邻叶共享，从而表示真实跳变。

当前动作粒度是：

```text
untie(all faces of one leaf)
```

它比逐面断开更保守，但实现和存储口径明确。逐面连通分量可作为未来压缩优化，不影响当前模型正确性。

## 6. 投影域界面选择

### 6.1 全共享模型

首先将所有叶转换为共享顶点 p1，并施加 hanging-node 约束，得到 `μ_tied`。

实验表明全域 C0 不可用：测试投影 PSNR 从 p5 的 `43.3745 dB` 降到 `41.4089 dB`。这证明工业衰减场不能被全局连续性替代。

### 6.2 Private-leaf 动作

对每个叶 `K`，候选动作恢复该叶在 p5 参考场中的一侧 p1 函数。对采样射线 `r`：

\[
\Delta p_{r,K}
=A_{r,K}(\mu_{\mathrm{ref},K}-\mu_{\mathrm{tied},K}).
\]

基于当前 tied 模型残差

\[
r_r=A_r\mu_{\mathrm{tied}}-y_r,
\]

预测单个动作造成的投影损失变化：

\[
\Delta L_K
\approx
\sum_{r\cap K}
\left(2r_r\Delta p_{r,K}+\Delta p_{r,K}^2\right).
\]

只有 `ΔL_K < 0` 的动作具有正预测收益。所有候选按同一 raw-byte 预算排序，不设置材料标签、空间区域、层级配额、最小分辨率或固定边界比例。

正式评分使用 `3,145,728` 条训练射线，相当于每个训练视角 32,768 条射线的总覆盖量。

### 6.3 当前预算与选择结果

正式选择配置：

```text
base nodal raw payload = 27,110,714 bytes
target raw budget      = 30,000,000 bytes
selected private leaves = 180,580
```

由于 private leaf 会改变共享连通关系，16 bytes/leaf 的初始成本模型略低估了实际增量。最终 raw payload 为 `30,635,076 bytes`。

被选 private leaves 的层级分布：

```text
L0: 10,743
L1: 61,177
L2: 105,598
L3: 3,062
```

这不是人工层级配额，而是投影收益排序的结果。

## 7. 精确前向和 CUDA 训练

### 7.1 射线积分

叶内三线性函数沿直线成为三次多项式，因此两点 Gauss-Legendre 积分可以精确计算每段积分。射线首先按真实叶边界切段，不进行稠密体采样。

### 7.2 约束梯度

前向流程：

```text
master physical values
  → sparse constraint gather/index-add
  → leaf-local p1 coefficients
  → inverse softplus logits
  → native Bernstein CUDA projector
```

反向梯度通过相同线性约束自动累加回 master/private vertices。

实现只缓存约束矩阵中的真实非零项，不再计算 `[local_corners, max_terms]` 中的空槽。4096-ray 单步时间由约 `7.9 s` 降到 `0.86 s`。

## 8. 固定结构联合精调

界面选择后冻结：

- octree topology；
- master/private vertex 关系；
- hanging-node constraints。

然后仅用 96 个训练投影优化所有 master/private values。最佳模型累计精调 720 步：

```text
20 steps  : batch 4096,  lr 1e-3
200 steps : batch 8192,  lr 5e-4
500 steps : batch 16384, lr 2e-4
```

不再修改 topology，也不使用 GT 或体域监督。

## 9. Compact nodal v4

部署文件包含：

- packed octree topology；
- float16 master/private attenuation values；
- private-leaf bitmask；
- level shapes、max degree 和必要 metadata。

不保存：

- optimizer；
- expanded leaf-local p1 coefficients；
- 完整 hanging constraint matrix；
- GT 或训练历史。

约束由 topology 和 private mask 确定性重建。当前 compact 文件：

```text
output/nodal_p1_96v_private30m_ft720/compact_nodal_v4.npz
size = 8,419,206 bytes
```

Compact 独立加载后的 `z=512` 切片 PSNR 为 `29.71712 dB`，checkpoint 为 `29.71719 dB`，差异约 `0.00006 dB`。

## 10. Viewer

Viewer 的 slice 和 projection 都调用 `forward_mu` 或精确 Bernstein projector：

- p1 按真实三线性函数求值；
- master/hanging/private 约束全部生效；
- 不把叶节点显示为单一平均衰减块；
- 当前 Viewer 直接加载 compact nodal v4。

地址：<http://127.0.0.1:8765/>

## 11. 当前正式结果

| 指标 | p5 | 当前 nodal p1 | 变化 |
|---|---:|---:|---:|
| Test projection PSNR | 43.3745 dB | 43.2965 dB | -0.0781 dB |
| Test projection MAE | 0.0018015 | 0.0019026 | +5.61% |
| 独立系数/master | 3,844,560 | 2,980,422 | -22.48% |
| Compact 文件 | 10,041,162 B | 8,419,206 B | -16.15% |
| 15-slice 平均 PSNR | 30.3990 dB | 30.7377 dB | +0.3386 dB |
| 15-slice 平均 MAE | 0.0129016 | 0.0124216 | 改善 3.72% |
| top-1% 梯度区 PSNR | 18.8976 dB | 19.1808 dB | +0.2832 dB |
| top-1% 梯度区 MAE | 0.0566187 | 0.0549285 | 改善 2.99% |
| 平均梯度幅值比 | 0.3860 | 0.3259 | 更平滑 |

结论：当前模型用更少的独立参数和更小的部署文件，获得了更高的切片值精度和高梯度区域值精度；代价是测试投影 MAE 略差，并且最尖锐界面的梯度幅值仍偏低。

## 12. 已验证但未采用的消融

### 12.1 全域 C0

- master vertices：1,218,241；
- compact：6.14 MB；
- test PSNR：41.4089 dB。

结论：压缩很强，但材料边界被错误连续化，不可采用。

### 12.2 更大 private 预算

简单增加 private leaves 的初始化指标不单调，因为多个动作共享射线，单动作加性近似忽略交叉项。联合精调后自由度更大的模型可以改善，但不是当前最佳率失真点。

### 12.3 Projection gain × interface jump

该排序将更多预算分给 L3，部分恢复梯度幅值，但降低总体投影和切片值精度。因此保留为消融，不作为主方法。

## 13. 已知问题

1. **Compact 首次加载慢**：2.02M 叶模型从 topology 重建约束约需四分钟。下一步应缓存紧凑 CSR 约束图。
2. **训练 checkpoint 很大**：展开的训练期约束 buffer 使 checkpoint 约 1.24 GB；正式部署文件不受影响。
3. **动作是叶级而非面级**：一个 private leaf 会断开其全部面，面级 connected-component action 可进一步节约参数。
4. **预算成本是近似值**：断开一个叶会改变共享连通关系，实际 master 增量不是严格常数。
5. **梯度幅值仍偏低**：虽然高梯度区域 PSNR/MAE 改善，但边缘强度比 p5 更低。
6. **依赖 p5 初始化拓扑**：当前不是从投影直接端到端生成 nodal 拓扑，而是在 p5 结果上执行函数空间重参数化和界面选择。
7. **稀疏视角信息上限仍存在**：p1 提供连续求值和亚单元定位，但不能恢复投影数据未观测到的频率。

## 14. 代码入口

- Nodal 表示与评分：`adaptive_ct/nodal.py`
- Bernstein 模型与 CUDA 参数桥接：`adaptive_ct/bernstein.py`
- Compact v4：`adaptive_ct/compression.py`
- 转换和精调：`scripts/run_nodal_p1.py`
- Private action 评分：`scripts/score_nodal_private.py`
- 多切片评价：`scripts/evaluate_nodal_slices.py`
- 单元测试：`tests/test_nodal.py`

完整测试结果：`100 tests passed`。

