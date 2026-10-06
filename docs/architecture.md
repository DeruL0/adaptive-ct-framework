# 代码框架

从多视角平行束投影重建物体的三维衰减系数场，存成不超过 8 MiB 的文件。只用投影数据：
不用传统重建结果做初值，GT 体只在训练结束后用于评测。

## 表示
- **八叉树**：物体所在立方体先均匀分成 48×48×48 个格子，任一格子可再一分为八（每条边对半切），
  最多到第 4 层（相当于 768³）。不再切分的格子叫叶子。
- **三线性叶**（当前主线）：存叶子 8 个角上的衰减值，叶内任一点的值是 8 个角值按位置的加权平均，
  权重非负且和为 1；所以只要存储值都 ≥ 0，整个场处处 ≥ 0。常数叶只存 1 个值（对照配置使用）。
  同一棵树可以同时有两种叶子（切分、常数→三线性转换都保持函数不变）。
- **存储**：直接存每个叶子的物理系数（半精度）+ 每叶 3 字节类型标记 + 每次切分 64 字节树结构；
  三线性叶 19 字节，常数叶 5 字节。预算按实际存储的字节计。
- **投影**：每个探测器像素对应一条直线，预测值是衰减系数沿直线的精确积分（按叶子边界分段计算）。
  场对系数是线性的，所以预测投影 = A·c，c 是全部系数排成的向量，A 的元素都 ≥ 0。

## 流水线
```
48³ 根网格，所有系数 = 训练投影平均值的一半
重复（每一轮）：
  1. 求解：树固定，求  min_{c≥0} ½‖A·c − y‖²                        (training/solve.py)
       误差 r = A·c − y；梯度 g = Aᵀ·r（把误差反投影回每个系数）
       每个系数的步长分母 d = Aᵀ·A·1（保证每步不会让目标变大）
       更新 c ← max(0, c − g/d)（小于 0 的置 0，即满足非负约束）
       加速：先把 240 个视角按角度交错分成 24 组、每组更新一次；之后用全部数据更新，
             并在"上一步移动的方向"上额外外推一段（外推点截断到 ≥ 0），方向变差就取消外推
       停止：最优性残差（在可行点上做一次上述更新，系数的相对变化）≤ 容差（每轮 1e-3，最后一轮 1e-4）；
             到遍数上限仍未达到则记为"未收敛"
       粗树用粗数据：最细叶子在探测器上还跨好几个像素时，把 2×2 或 4×4 个像素取平均当一条射线求解
             （射线数少 4 或 16 倍）；最后一轮总是用全分辨率
  2. 打分：在解好的模型上，把每条射线的残差平方按穿过长度分给路径上的叶子      (refinement/estimator.py)
  3. 选一批：按分数从高到低取，直到这批的分数之和占总分的一半，不超出字节预算   (refinement/marking.py)
  4. 切分：被选中的叶子一分为八，子叶系数精确复现父叶函数，所以投影不变；
           下一轮求解从这里继续                                              (refinement/controller.py)
直到：字节预算用完 / 没有可切的叶子 → 最后一轮按更多遍数严格求解
导出：系数四舍五入到半精度（不再重新拟合），评测，写入紧凑文件
```

## 目录
```
adaptive_ct/
  core/            config.py  geometry.py  projector.py(CUDA 投影器封装)
  data/            dataset.py  splits.py  rays.py
  representation/  octree.py  factory.py
  evaluation/      metrics.py  render.py  volume.py
  codec/           budget.py(字节计数)  quantize.py(半精度舍入)  artifact.py(紧凑文件)
  refinement/      estimator.py  marking.py  controller.py
  training/        loop.py  solve.py  export.py  validate.py  __main__.py
  viewer/          server.py + web/
native/            bernstein_octree.cu(八叉树精确积分)  projector.cu(稠密体投影，仅查看器用)  ext.cpp
configs/stage7/    p1.yaml（主线）  p0.yaml（常数叶对照）
scripts/audit/     gate_chain.py：离线诊断（用到 GT，只用于分析，不进入流水线）
tests/             与 adaptive_ct/ 镜像
```
依赖只自上而下：core ← data ← representation ← evaluation ← codec ← refinement ← training；viewer 只读复用下层。

## 视角划分
- 训练 240 个视角：求解和打分只用这些。
- 检验 16 个视角：每轮报告投影误差，不参与任何决策。
- 测试 4 个视角：只在最后评测。

## 运行
```powershell
.venv\Scripts\python.exe -m adaptive_ct.training --config configs\stage7\p1.yaml
scripts\test.ps1
```
环境：项目内 `.venv`（Python 3.10，torch 2.11.0+cu130，匹配本机 CUDA 13.0，可重编原生扩展）。

## 已知缺口
- 打分规则仍是"残差平方按长度分摊"，定位不准（同一份数据，选出的树比最优树差约 1.4 dB）。
- 主流程还不在常数叶 / 三线性叶之间做选择（表示和存储已支持混合）。
- 保边正则（在检验视角上选强度，候选包含 0）尚未实现。
- 一遍全数据（2.5 亿条射线，正投影 + 反投影）约 12 秒（反投影内核改为线程束内先合并再原子累加，
  原为 35 秒）；其中反投影约占 2/3、正投影约 1/3。

只保留当前方法：旧的训练方式、旧的压缩编码、采样积分、高阶多项式叶等都已从代码中删除，
历史版本只在 git 记录里（标签 `legacy-adam-pipeline` 和此前的提交）。
