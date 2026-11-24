# Backbone 前裁剪修订验证

修订从 `加速@862d9fcc7e14b9ef8f6b9f1479df469251c3302d` 开始。已核对 `d8082d2`、`48fe04a` 和 `be1601e` 都是祖先，完整运行原入口帮助和两个权重的 SHA-256 检查。真实帧、静态图和服务器试验使用独立工作树。

## 真实帧与回归

KITTI `000005_10`，原图 375×1242，E20 `fixed_56`、V2、FP32、score 0.40、top-K 500、同类 paired NMS 0.50。Windows 本机 CPU 使用 OpenStereo `23d71c9` 和 TIMM 0.4.12。

| 裁剪行数 | 实际入口 NCHW | 共享 Backbone 实际 NCHW | NMS 前 / 后 / iDisp |
| ---: | --- | --- | ---: |
| 0 | 1×3×375×1242 | 2×3×384×1248 | 118 / 24 / 24 |
| 64 | 1×3×311×1242 | 2×3×320×1248 | 117 / 24 / 24 |
| 96 | 1×3×279×1242 | 2×3×288×1248 | 114 / 26 / 26 |
| 128 | 1×3×247×1242 | 2×3×256×1248 | 99 / 24 / 24 |

每档 Backbone 调用一次，NMS-on 调用一次，iDisp 入口候选 ID 与 NMS 保留 ID 同序。全图 NMS-off 的前置候选 ID 与 on 相同，118 对全部进入 iDisp。0、128 行的新入口与 `862d9fc` 基线的 pair ID、类别和有效性完全相同。128 行的新旧入口之间也一致。框、得分、视差和 RoI 原点按 `atol=1e-5, rtol=1e-5` 校验。合成行索引测试对四档的两目归一化像素和底部补齐后的第一层输入做逐元素验证。

静态图通过了四档 V2 和 128 行 V3-Small 的 ONNX 查询与 ONNX Runtime CPU 核对。八个前端输出与 PyTorch CPU 在真实帧上的最大绝对差为 V2 `1.27e-4`、V3-Small `5.01e-6`。128 行静态图会拒绝 375 行全图和 119 行二次裁切。TensorRT 包装器单测同样确认错误 NCHW 在 buffer 操作前被拒绝。实际 TensorRT 引擎构建、层精度检查与板端计时留待目标环境。

## 运行入口

- `scripts/infer.py` 默认从校正像素先同裁再归一化。原入口兼容对照使用 `--input-path model`。用 `--expect-commit` 锁定运行提交，用 `--trace-json` 保存实际加载模块、权重 SHA、NMS 调用、候选 ID、Backbone 和 iDisp 形状。
- `scripts/evaluate_edge_acceleration.py` 接入同一预处理，保存每帧前后 ID、GT 损失路径、双目 IoU、候选得分和 RoI 去留。C 分支的 Car 专用权重输出 C0 对齐后的共同有效前景像素、掩码 IoU 和视差差值，供离线核查。
- `scripts/benchmark_edge_acceleration.py` 将读图、CPU 裁剪、归一化、H2D、模型前向和 D2H 分开记录。分段插桩独立运行。用 `--compare-input-paths` 在同帧同轮内交替两个裁剪入口，按帧归组计算端到端 P95 差值区间。
- `scripts/export_edge_frontend.py` 记录两目静态 NCHW binding、Backbone 类型和导出权重。V3 原型使用 `--backbone v3-small --seed 17`。TensorRT 入口同时校验 manifest、权重与引擎哈希、绑定形状和每次实际输入。paired NMS 使用引擎外原实现。

## C1 复核和发布

原 Jetson 单帧的 `3707 / 3554 / 3401` 是待人工确认的同车正例候选。固定反例为 `000198_10:1` 与 `000198_10:5`，普通 NMS 后分别匹配两个 GT，在旧 C1 规则下可出现 100% 双目包含。几何 C1 保持诊断用途和默认关闭。独立遮挡复核集、目标板延迟和上线门槛仍按实验计划执行。
