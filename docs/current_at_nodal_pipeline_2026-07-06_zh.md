# 稀疏视角工业 CT：自适应 Nodal-p1 自由不连续重建 Pipeline

更新日期：2026-07-06

本文描述仓库当前实际实现的算法、当前已接受模型的真实来源，以及尚未完成验证的部分。旧的 p5、coarse-to-fine 和 AT Stage-A 文档只作为历史实验记录，不代表当前主流水线。

## 1. 目标和约束

目标是在有限投影视角下同时实现：

1. 较高的投影和体域重建精度；
2. 对裂纹、孔隙、薄壁和多材料界面的显式不连续表达；
3. 对均匀及平滑区域使用共享的局部三线性函数降低参数量；
4. 导出可直接部署和渲染的紧凑模型。

当前正式路径只使用扫描投影和几何进行训练与结构选择。GT 体仅用于最终评价，不参与优化、相场求解、split/merge/untie 或超参数选择。

当前没有使用：

- support mask 或材料标签；
- 强制最低分辨率或逐层比例；
- TV、continuity 或 Sobolev 后处理；
- 外部训练集；
- GT 监督。

## 2. 数据协议

当前 v4 结果使用：

- 96 个训练视角，均匀覆盖 `[0, π)`；
- 4 个独立测试视角；
- `1024 × 1024` 探测器；
- `473 × 1024 × 1024` GT 体，仅用于评价；
- 平行束投影几何。

配置文件：

```text
configs/research/industrial_detail_v5_96v_equal_ratio_anisotropic.yaml
```

当前采用各轴近似等压缩率的各向异性八叉树：

```text
L0 = [22,  48,  48]
L1 = [44,  96,  96]
L2 = [88, 192, 192]
L3 = [176, 384, 384]
```

## 3. 当前总体流程

```text
96-view projection images + scan geometry
    |
    v
h-adaptive projection reconstruction
    |
    v
balanced octree + local Bernstein p1 field
    |
    v
conforming nodal conversion
    |-- smooth interfaces: shared master vertices
    |-- coarse/fine interfaces: hanging-node constraints
    `-- discontinuous interfaces: private leaf-face corners
    |
    v
AT phase-field solve on the same octree
    |
    |-- leaf phase score  -> h-split / merge proposals
    `-- face phase score  -> tie / untie proposals
    |
    v
2:1 balance + rebuild nodal constraint graph
    |
    v
fixed-topology projection-domain fine-tuning
    |
    v
total-energy decision + RSS/CV safety guard
    |-- accept: replace current compact state
    `-- reject: restore previous accepted state
    |
    v
epsilon continuation: 8h -> 4h -> 2h
    |
    v
compact nodal v5 + function-aware Viewer
```

## 4. 衰减场表示

### 4.1 每个叶节点都是局部 p1 函数

叶节点 `K` 内使用三线性 Bernstein 函数：

\[
\mu_K(u,v,w)=
\sum_{i,j,k\in\{0,1\}}
c_{ijk}B_i(u)B_j(v)B_k(w).
\]

八个角点系数通过 softplus 保持非负。均匀区域不需要单独的 p0 建模分支：当八个角点值相等时，p1 自然退化为常数。

### 4.2 Nodal master values

实际优化变量不是每叶独立的八个系数，而是 master nodal values。每个叶的局部角点值由稀疏约束矩阵展开：

\[
c_{K,q}=\sum_m W_{K,q,m}v_m.
\]

其中：

- `v_m` 是独立 master value；
- `W` 表示共享顶点或 hanging-node 插值关系；
- 平滑同层叶共享同一几何顶点；
- 梯度通过相同约束自动累积回 master values。

### 4.3 2:1 balance 和 hanging nodes

相邻叶层级差限制为不超过一级。粗细交界的非独立节点满足：

- 粗边中点：两个粗端点各 `0.5`；
- 粗面中心：四个粗角点各 `0.25`；
- tied 粗细界面严格满足 C0，不依赖 continuity loss。

## 5. Voxel level 与 face level

Voxel/leaf topology 决定：

- 单元的位置和尺寸；
- 局部 p1 函数的定义域；
- 可执行的 split/merge；
- 射线与单元的真实交段。

每个叶另外保存六个有向 face bits：

```text
[x-, x+, y-, y+, z-, z+]
```

普通 face 的角点参与共享和 hanging-node 约束。Private face 的四个叶局部角点变为私有 occurrence，从而允许该叶一侧与邻叶产生衰减跳变。

当前实现的重要边界：

1. face mask 是 `[leaf_count, 6]` 的 leaf-side 状态，不是唯一 physical-face 图；
2. 标记一个 face 会私有化该面的四个叶角点，这些角点也属于其他相邻面；
3. 因此它是 face-aware 原型，但还不是严格只切断一个物理面的有限元连通图；
4. 当前接受的 v4 模型来自旧的 whole-leaf private mask，加载时每个 private leaf 被展开为六个 private faces；
5. v4 的新 face-level 候选均被总能量判据拒绝，因此当前 artifact 尚未真正采用选择性的逐面 untie。

## 6. 精确前向投影

射线首先按真实八叉树叶边界切段。p1 函数沿直线参数变成三次多项式，因此两点 Gauss-Legendre 积分可以精确计算每个段积分。

前向路径为：

```text
master values
  -> sparse nodal constraint expansion
  -> leaf-local p1 coefficients
  -> packed octree ray traversal
  -> exact segment integration
  -> projection value
```

Slice 和 Viewer 逐点调用同一个 p1 场求值，不把叶显示成单一平均衰减块。Projection 使用同一组展开后的局部 p1 系数。

## 7. Ambrosio-Tortorelli 相场

训练期在同一棵 balanced octree 上维护相场 `v`：

- `v ≈ 1`：平滑区域；
- `v ≈ 0`：候选不连续区域。

当前体积分为：

\[
E_{\mathrm{AT}}(\mu,v)=
\alpha\int_\Omega (v^2+\eta)|\nabla\mu|^2\,dx
+
\beta\int_\Omega
\left(
\epsilon|\nabla v|^2+
\frac{(1-v)^2}{4\epsilon}
\right)dx.
\]

其中：

- `η=10^{-6}` 保持残余椭圆性；
- 体积分在每叶 `2×2×2` Gauss 点计算；
- 叶按物理体积抽样；
- 固定 `μ` 时，`v` block 使用 matrix-free CG；
- 固定 `v` 时，`μ` 使用投影数据项和 AT 项联合更新。

为了让相场看到 private nodal face 两侧的跳变，另外计算 skeleton face term：

\[
\int_F \bar v^2[\mu]^2\,dS.
\]

该项包含同层和粗细层交界，并按共享面积归一化。它补充了叶内体积分无法观察 private-face 跳变的问题。

## 8. 相场驱动的结构动作

### 8.1 Leaf score

对 Q1 相场，叶内最小值出现在角点，因此使用：

\[
s_K=1-\min_{q\in K}v_q.
\]

- 高分且未到最高层：提出 h-split；
- 低分的完整八兄弟组：可以进入 merge 检查。

### 8.2 Face score

每个叶面使用：

\[
s_{K,F}=1-\min_{q\in F}v_q.
\]

高分提出 untie，低分提出 retie。每轮只允许有限数量的 face 状态变化，形成 trust region，避免一次性把全部界面打开或关闭。

### 8.3 Split

父叶分裂为八个同 degree 子叶：

- p1 函数通过 Bernstein subdivision 保持；
- 父节点外侧 private face 只传给接触该外表面的子叶；
- 新产生的八子内部面默认 tied；
- 随后执行 2:1 balance。

### 8.4 Merge

八兄弟只有同时满足以下条件才可合并：

1. 相场 leaf score 低于 merge threshold；
2. 八子内部没有 private cut；
3. 八个子 p1 能被一个父 p1 近似；
4. coefficient-space compatibility error 不超过阈值。

父节点外侧 face 状态由接触该外表面的子面继承。

## 9. 完整可逆循环

每个 epsilon round 执行：

1. 从最后一个 accepted compact state 重新加载模型；
2. 交替更新 `μ` 和求解 `v`；
3. 计算 leaf/face phase scores；
4. 提出 split、merge、untie、retie；
5. 执行动作并重建 2:1 nodal constraint graph；
6. 前200步只更新动作两环邻域内的 master values，随后300步全局精调；
7. 在固定训练射线样本上计算唯一总能量；
8. 总能量下降且通过宽松 RSS/CV 灾难护栏时接受；
9. 否则丢弃候选并恢复上一 accepted compact state；
10. 继续下一 epsilon。

当前 continuation：

```text
epsilon = 8 h_finest
       -> 4 h_finest
       -> 2 h_finest
```

`2h` 是当前最锐利的最终相场阶段。

## 10. 总能量验收

当前唯一验收能量为：

\[
E_{\mathrm{total}}=
\frac{\mathrm{RSS}}{\sigma^2}
+\alpha E_{\mathrm{smooth}}
+\beta_{\mathrm{eff}}E_{\mathrm{edge}}
+k_{\mathrm{nodal}}\log n.
\]

其中：

- `σ²` 从最低信号5%训练探测器像素估计一次并冻结；
- `E_smooth` 为 AT 叶内平滑项与显式 face-jump 项；
- `E_edge` 为相场梯度项和双井项；
- `beta_eff = c·σ²·log(n)`，当前 `c=1`；
- `k_nodal` 只统计独立 master values，不重复对 face bits 计费。

接受条件只有 `E_total(candidate) < E_total(accepted)`。另外保留不参与排序的灾难护栏：

\[
\mathrm{RSS}_{candidate}
<
\mathrm{RSS}_{accepted}(1+3\,\mathrm{CV}).
\]

CV 从固定训练射线样本的分块 RSS 估计。Test views 不参与任何选择。

## 11. Compact nodal v5

正式导出包括：

- packed hierarchy；
- leaf degree；
- float16 master/private nodal values；
- per-leaf 六面 private bitmask；
- level shapes 和必要 metadata。

不导出：

- optimizer；
- 相场 `v`；
-训练历史；
- 展开的每叶八系数；
- 完整 constraint matrix。

加载时从 octree topology 和 face mask 确定性重建 nodal constraint graph。

当前 artifact：

```text
output/at_adaptive_96v_v4_total_energy_relcg/compact_nodal_v5.npz
```

## 12. 当前正式结果

| 指标 | 当前 v4 |
|---|---:|
| Test projection PSNR | 44.02025 dB |
| Test projection MAE | 0.0018306 |
| Slice PSNR | 29.89614 dB |
| Slice MAE | 0.0116969 |
| High-gradient PSNR | 18.56654 dB |
| High-gradient MAE | 0.0457587 |
| Gradient ratio | 0.19741 |
| Near-air halo MAE | 0.0182299 |
| Leaf count | 2,025,290 |
| Independent nodal coefficients | 2,926,140 |
| Private face bits | 1,059,378 |
| Compact file | 8,304,565 bytes |
| Raw compact payload | 32,163,427 bytes |

相对 `473×1024×1024 float32` 稠密体，实际 compact 文件约压缩 239 倍。

层级分布：

```text
L0:   11,864
L1:  187,803
L2:  861,839
L3:  963,784
```

## 13. v4 结构轮次的真实结果

每轮最多执行64 split、64 merge和128 face changes；全部动作相隔至少两环。三轮安全护栏均通过，但总能量均上升：

| epsilon | Split | Merge | Face | ΔRSS/σ² | Δcomplexity | ΔE_total | 接受 |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| `8h` | 0 | 64 | 128 | `+5,555,121` | `-1,419` | `+5,553,702` | 否 |
| `4h` | 0 | 64 | 128 | `+5,494,058` | `-2,248` | `+5,491,810` | 否 |
| `2h` | 0 | 64 | 128 | `+5,537,285` | `-1,806` | `+5,535,479` | 否 |

AT 平滑项和边缘项都轻微下降，但加权变化小于 `1e-8`，不足以补偿投影 RSS 增长。RSS 安全护栏不是拒绝原因；拒绝来自唯一总能量法官。

相场在 `4h` 和 `2h` 达到相对残差 `<1e-4`，`8h` 在200步上限时为约 `6.3e-4`。由于 `c=1` 下相场最小值仍较高，本次没有产生 split 候选。

因此最终 v4 保留原 Stage-C 函数和拓扑。结果表明当前提案质量不足，而不是旧 fidelity gate 造成零接受率。

## 14. 当前结论

已经成立：

- packed anisotropic octree；
- nodal p1 共享函数场；
- 2:1 hanging-node 约束；
- private-face 数据结构；
- 显式跨叶 face-jump AT 项；
- 相场驱动 split/merge/untie/retie；
- epsilon continuation；
- 局部两环加全局投影重训、总能量验收和 RSS/CV 灾难护栏；
- compact v5 和函数化 Viewer。

当前主要问题：

1. 高频恢复不足：`gradient ratio=0.197`，边缘幅值明显偏低；
2. `8h` 相场 CG 在200步内仍未达到相对残差 `1e-4`；
3. v4 已给每轮500步消化，但候选 RSS 仍系统性恶化；
4. 当前 accepted artifact 没有接受任何 AT 结构动作；
5. 当前 face mask 是 leaf-side corner privatization，不是唯一 physical-face connectivity；
6. face bit 可能重复存储同一物理面的两侧状态；
7. private face 会连带影响其四个角点关联的其他面；
8. face-level 只能在现有叶边界放置跳变，叶内斜界面仍依赖 h-refinement；
9. Viewer 的 config report 可能显示旧训练指标，当前结果应以 v4 `report.json` 为准。

因此当前方法最准确的定位是：

> 已实现并验证安全回滚的自适应 nodal-p1 自由不连续原型；压缩率和投影拟合较强，但高频细节保持及严格 physical-face 拓扑仍未达到最终论文方法的完成度。

## 15. 代码和结果入口

- Nodal 约束构造：`adaptive_ct/nodal.py`
- Bernstein octree 与 projector bridge：`adaptive_ct/bernstein.py`
- AT 体积项、face jump 和相场 CG：`adaptive_ct/ambrosio_tortorelli.py`
- Split/merge/face 动作：`adaptive_ct/at_adaptive.py`
- Compact v5：`adaptive_ct/compression.py`
- 完整自适应运行：`scripts/run_at_adaptive.py`
- 当前报告：`output/at_adaptive_96v_v4_total_energy_relcg/report.json`
- 当前 compact：`output/at_adaptive_96v_v4_total_energy_relcg/compact_nodal_v5.npz`
- Viewer：http://127.0.0.1:8765/
