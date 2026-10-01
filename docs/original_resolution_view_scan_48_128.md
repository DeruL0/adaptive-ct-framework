# Figure473 原分辨率 48/128 视角实验

## 协议

- GT：`473×1024×1024`
- detector：`1024×1024`
- train views：48 或 128，均匀覆盖 `[0, 2π)`
- test views：两组共用同一组 4 个角度，并且不与任一训练角重合
- 模型：`[96,192,384,768]` packed Bernstein hierarchy
- 训练：15,000 steps、blend50 weighting、continuity、相同 h/p milestones、7.5M leaf budget
- 为保持每个视角的期望射线访问次数与原 16-view 配置一致，batch size 按视角数缩放：
  - 48 views：24,576 rays/step
  - 128 views：65,536 rays/step
- refinement residual 使用全部训练视角，每个视角 32,768 rays。
- 体积指标：模型和原始 GT 统一采样到 `384³` voxel-centre 网格；材料阈值为 `0.1`。

## 结果

| Train views | Test projection PSNR | Full-volume PSNR | Full-volume MAE | Material PSNR | Material MAE |
|---:|---:|---:|---:|---:|---:|
| 48 | **36.7637 dB** | 31.0691 dB | 0.012652 | 23.0414 dB | 0.043787 |
| 128 | 36.3200 dB | **32.1974 dB** | **0.011293** | **24.5666 dB** | **0.033777** |

128 相对 48：

- full-volume PSNR：`+1.1284 dB`
- full-volume MAE：`-0.001358`
- material PSNR：`+1.5252 dB`
- material MAE：`-0.010010`
- test projection PSNR：`-0.4437 dB`

投影 PSNR 与体积指标的排序相反。48-view 在 4 个测试投影上更高，但 128-view 的真实三维体积和材料区明显更准确。这再次说明不能用投影 PSNR 代替体积质量。

## 饱和判断

128-view 尚未达到体积质量饱和。48→128 仍带来超过 1 dB 的 full-volume 收益和超过 1.5 dB 的材料区收益，不能把 48-view 当作足够视角数，也不能声称最小充分视角数已经找到。

在用户限定只跑 48/128 的前提下，可以得到的严格结论是：**该物体和当前方法所需的饱和视角数大于 128，或至少没有证据表明 128 已饱和。**

## 仍未解决的空间分配问题

更多视角改善了体积系数估计，但没有修复 h-refinement 的根本分配偏差：

- 48-view：GT root mean >0.1 的材料核心单元仍停留在 L0 的比例为 100%。
- 128-view：同一比例仍为 100%。

因此视角增加和空间分配修复是两个独立问题。128-view 的体积提升证明更多观测有效，但下一步仍应修改材料内部最低分辨率规则。

