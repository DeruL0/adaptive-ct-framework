# 当前 16 视角 CT 重建流水线

本文对应最终配置 `configs/figure473_dicom_orig_accuracy_final.yaml` 和 2026-06-29 的真实 15,000 步测试。

## 1. 数据和约束

- 16 个训练视角，4 个测试视角，探测器原始分辨率 1024×1024。
- test views 只做最终评价，不参与 h/p refinement。
- 训练目标是投影域 WLS；不加载 dense ground-truth volume，不启用传统 TV。
- 并行束射线由 `adaptive_ct/geometry.py` 生成，CUDA 直接积分 sparse Bernstein leaves。

## 2. 任意深度 packed Bernstein octree

层级分辨率为 `[96, 192, 384, 768]`。拓扑不再创建每层稠密三维索引，而是保存：

- `node_child_base:int32`：内部节点指向连续 8 个子节点，叶节点为 `-1`；
- `node_leaf_id:int32`：叶节点到系数块的映射；
- 每个 leaf 的 level、cell coordinate、degree、coefficient offset。

CPU 和 CUDA 都从 96³ roots 沿 packed hierarchy 向下遍历。CUDA forward、backward 和 ray-segment attribution 使用同一套任意深度查找。residual segment chunk 根据最高层和 `residual_max_buffer_mb=512` 动态计算，不分配 768³ 索引。

h-split 后执行 face-adjacent 2:1 balance，并受 `max_leaf_count=7,500,000` 硬预算约束。

## 3. 投影损失 schedule

每个训练 iteration 解析当前投影权重。最终实验使用：

- iteration 1–6000：inverse weighting，先约束低衰减和背景一致性；
- iteration 6001–15000：inverse/target-power blend，兼顾全局误差和材料边界。

训练 loss、residual attribution 和梯度 R-D p-score 都复用同一 iteration 对应的权重参数。

## 4. h-refinement

早期 bootstrap：

- iteration 80：L0→L1 coefficient split；
- iteration 180：L1→L2 coefficient split。

后期真实 residual refinement：

- iteration 9000：L2→L3；
- 使用全部 16 个训练视角，每个视角固定抽样 32,768 rays；
- 对每条射线把加权投影残差按 leaf line-integral contribution 分配；
- 选择 top fraction 与 `score >= 0.5 × positive-score q95` 的并集；
- L2→L3 最大选择比例 10%，再执行 2:1 balance 和总 leaf 预算检查。

WPL3 run 在 L3 激活 475,624 leaves。匹配消融显示它没有优于 WP，因此正式配置关闭 L3；该实现和实验结果仍保留。

## 5. 跨层 coefficient continuity

continuity 从 iteration 6001 起，在 1000 步内线性升到 `0.001`：

- edge-preserving Huber，不使用原始 L2；
- 同时处理同层 face 和粗细层 face；
- 在共享 face 上做二阶 quadrature，比较完整 Bernstein 面多项式采样值；
- 按共享面积归一化；
- topology 变化时才重建 face-pair cache。

## 6. 纯梯度 rate-distortion p-refinement

iteration 10500 对 L2/L3 执行 `[1,1,1]` elevation：

1. 用 16-view residual 选 top 8% 候选池；
2. 临时做保持函数不变的 degree elevation；
3. 构造完整 elevation matrix `E`；
4. 把 logit gradient 转换到物理 Bernstein coefficient gradient；
5. 将梯度投影到旧函数空间的正交补，剔除旧自由度贡献；
6. 用覆盖全部训练角度的 8 个 ray batches 累积梯度；
7. 计算 `S = eta * ||Q_new^T g||² / delta_R`，其中 `delta_R` 是 float16 新增字节；
8. 最多接受 eligible leaves 的 2%，新增系数不超过 1M；
9. 重建 optimizer，剩余训练全局联合收敛。

正式 WP run 接受 11,890 个 L2 leaves，新增 83,230 个系数。

## 7. 导出和 viewer

`compact_bernstein_octree_packed_v3` 保存 packed nodes、任意层 leaves、混合 degree 和量化系数。v1/v2 仍可读取。

viewer 加载 checkpoint 或 v3 artifact 后，投影视图调用模型的真实 ray integrator，不是体块截图。volume texture 仅用于交互预览，不参与训练和正式投影指标。

## 8. 最终真实测试

正式结果：`output/figure473_dicom_orig_accuracy_controlled_wp`。WPL3 对照保存在 `output/figure473_dicom_orig_accuracy_wpl3_blend50_l3_10`。

| 指标 | float16 baseline | float16 final | 结果 |
|---|---:|---:|---|
| full-test PSNR | 32.3618 dB | 32.9471 dB | 通过 |
| full-test MAE | 0.006929 | 0.006357 | 通过 |
| high-gradient top-1% PSNR | 22.9380 dB | 23.4126 dB | +0.4746 dB，未达到 +0.5 dB |
| gradient-ratio distance | 0.56791 | 0.51176 | 改善 9.89%，未达到 15% |

正式 WP 运行时间 141.47 秒；峰值 allocated/reserved GPU memory 为 1.02/1.55 GiB。三层 degree-0 CUDA step 从 0.4918 ms 变为 0.5188 ms，回退 5.48%，在 10% 上限内。

最终 uint8 相对 float16 下降 0.0788 dB，高梯度 MAE 未恶化，因此量化决策是保留 uint8。完整机器可读结论在正式结果目录的 `acceptance_report.json`；其中两个结构验收项明确保留为失败。

严格匹配消融如下。三组只改变 p/L3 开关；seed、权重、continuity、训练步数和测试集完全相同。

| variant | leaves | coefficients | full PSNR | high-gradient PSNR | gradient-ratio distance |
|---|---:|---:|---:|---:|---:|
| W | 1,711,142 | 1,711,142 | 32.94457 | 23.41028 | 0.51172 |
| WP | 1,711,142 | 1,794,372 | 32.94711 | 23.41258 | 0.51176 |
| WPL3 | 2,331,874 | 2,427,662 | 32.94735 | 23.40898 | 0.51259 |

L3 相对 WP 多 620,732 leaves，只增加 0.00024 dB full PSNR，并使两个结构指标变差，因此按真实 test Pareto 规则拒绝 L3。

## 9. 代码入口

- packed hierarchy：`adaptive_ct/bernstein.py`
- CUDA ABI：`adaptive_ct/backend.py`
- CUDA kernels：`native/bernstein_octree.cu`
- schedule、h/p refinement：`adaptive_ct/train.py`
- cross-level continuity：`adaptive_ct/losses.py`
- v3 export/load：`adaptive_ct/compression.py`
- 量化评价：`adaptive_ct/quantization_eval.py`
- 验收报告：`adaptive_ct/accuracy_acceptance.py`
- viewer server：`adaptive_ct/viewer/server.py`

## 10. B2：48-view hp + blend50 验证

为分离 16-view 欠采样和 hp 表示本身的影响，使用 `figure473_ctgs96_48v` 做了严格匹配实验：48 个训练视角、12 个独立测试视角、96×96 detector、15,000 步。两组使用相同 seed、packed `[24,48,96]` h 拓扑、blend50 schedule 和 continuity；hp 组只在 iteration 10500 增加规划内的 top-2% 梯度 R-D p-refinement。

| variant | leaves | coefficients | float16 full PSNR | high-gradient PSNR | gradient-ratio distance |
|---|---:|---:|---:|---:|---:|
| h-only | 26,844 | 26,844 | 34.38986 | 28.17644 | 0.136594 |
| hp | 26,844 | 28,139 | 34.38946 | 28.15010 | 0.136428 |
| hp8 stress | 26,844 | 32,031 | 34.38647 | 28.13272 | 0.136043 |

hp 接受 185 个 L2 leaves，新增 1,295 个系数。相对 h-only：full PSNR `-0.00039 dB`，high-gradient PSNR `-0.02634 dB`，gradient-ratio distance 仅改善 `0.000165`。

为排除 2% 预算过小，又执行 hp8 压力测试：提升全部 top-8% 候选，共 741 leaves、5,187 个新增系数。相对 h-only，full PSNR `-0.00339 dB`，high-gradient PSNR `-0.04372 dB`，gradient-ratio distance 只改善 `0.000551`。因此在视角充分和更大 p 预算下，当前 degree-1 梯度 R-D p 路线仍没有可测的 held-out 泛化收益；这同时排除了“16-view 统计功效不足”和“2% 预算过小”两个解释。机器报告位于 `output/b2_figure473_48v_hp_blend50/hp_validation_report.json` 与 `output/b2_figure473_48v_hp8_blend50/hp_validation_report.json`。
