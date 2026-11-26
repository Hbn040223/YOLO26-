# Jetson Orin NX 原生 PyTorch 5 FPS 实测

## 结论

当前最快的纯 PyTorch 配置在 120 秒内完成 **569/600 帧（4.74 FPS）**，窗口结束时积压 **31 帧**，未达到 5 FPS。9–10 对框场景的单帧处理 P95 为 **336–357 ms**，整段回放含排队的 P95 为 **6.31 秒**。原生 5 Hz 的验收要求是 120 秒内完成 600 帧、零积压、零丢帧，且含排队 P95 ≤ 200 ms。

## 测试条件和口径

- Jetson Orin NX，25 W 模式，GPU 频率上限 408 MHz。PyTorch 2.8.0、CUDA 12.6。TensorRT 对照使用 10.3.0。
- MobileNetV2 0.5×，权重为 `outputs/v2-050/formal/best_r2.pt`，SHA-256 为 `dd9990d5435d5da9696437009449bda4713d5d41a196a2d495e7363a3abd8095`。完整双目图，score 0.48，同类双目 NMS IoU 0.50，top-K 500，15 m 粗筛，10 m 精测。iDisp 使用 `fixed_56` 和 33 个视差档。
- 四帧循环序列固定为 KITTI 2015 测试帧 `000000_10`，以及训练帧 `000004_10`、`000005_10`、`000058_10`。原生回放中这四帧分别有 0、9、10、2 对框进入 iDisp。
- **单帧处理**从已解码的内存图像开始，包含归一化、传输、推理和结果回传。**含排队延迟**从预定到达时间算起。**GPU 预载前向**使用已经归一化且传入 GPU 的图像。三种口径对应不同计时区间。

## 原生回放

两轮原生回放均使用 AMP、cuDNN TF32 和 `bmm`，输入频率均为 5 Hz。第二轮加入前端 channels-last 和 GPU 归一化。

| 配置 | 120 秒内完成 | 实际吞吐 | 窗口末积压 | 丢帧 | 含排队 P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AMP + cuDNN TF32 + `bmm` | 557/600 | 4.64 FPS | 43 | 0 | 8.95 s |
| 加 channels-last 和 GPU 归一化 | **569/600** | **4.74 FPS** | **31** | **0** | **6.31 s** |

| 每帧 iDisp 框对 | 第一轮单帧 P50 / P95 | 第二轮单帧 P50 / P95 |
| ---: | ---: | ---: |
| 0 | 132 / 167 ms | 118 / 168 ms |
| 2 | 193 / 252 ms | 185 / 243 ms |
| 9 | 251 / 319 ms | 249 / 336 ms |
| 10 | 261 / 330 ms | 261 / 357 ms |

第二轮多完成 12 帧，轻负载处理的中位数下降，9–10 对框的 P95 上升。现有两轮记录无法归因这一尾延迟变化。首轮独立分项测量中，9–10 对框的 iDisp CUDA 区间 P50 约为 105–117 ms。原生前端 CUDA Graph 保持四帧输出一致，四帧平均处理时间为 196 ms，普通执行为 191 ms。

## 批量矩阵乘法的独立收益

`bmm` 将逐视差档计算的左右特征相关性改为按扫描行批量矩阵乘法，后续局部聚合结构保持原实现。用 `--idisp-cost-backend bmm` 可在推理、评估和回放中开启，默认使用 `loop`。同帧、同精度的严格 FP32 对照关闭了 CUDA matmul 和 cuDNN 的 TF32。

| 帧 | iDisp 框对 | 端到端 `loop` P50 / P95 | 端到端 `bmm` P50 / P95 | iDisp P50，`loop` → `bmm` |
| --- | ---: | ---: | ---: | ---: |
| `000000_10` | 0 | 338 / 389 ms | 332 / 424 ms | 0 → 0 ms |
| `000004_10` | 9 | 535 / 603 ms | **481 / 572 ms** | 161 → 98 ms |
| `000005_10` | 9 | 551 / 616 ms | **504 / 584 ms** | 157 → 103 ms |
| `000058_10` | 2 | 433 / 526 ms | 456 / 533 ms | 54 → 34 ms |

两张 9 对帧的端到端 P50 分别降低 **54 ms（10%）** 和 **46 ms（8%）**。严格 FP32 对照中四帧的 iDisp 入口和最终输出 ID 一致，最大测距变化小于 0.000001 m。固定框 AMP 复测中，两张 9 对帧的缓存 RoI iDisp P50 分别从 **148 → 96 ms**、**156 → 99 ms**，2 对帧从 **48 → 33 ms**。零框帧确认没有 iDisp 调用。 [早期固定框和代价体原始记录](../outputs/v2-050/idisp-fpn/jetson/20261126-fixed-pairs/fixed_pairs.json)

### 历史前向速度的对照

早期报告中的 165 ms 来自 `000000_10` 零框对帧的 GPU 预载前向。后续同帧交替复测，每次只切换 cuDNN TF32，预载前向 P50 在关闭时为 **307 ms**、开启时为 **166 ms**。9 对帧 `000004_10` 的同口径结果为 **486 → 340 ms**。看 `bmm` 的收益应使用同精度的 `loop`／`bmm` 端到端表。 [历史零框对帧计时](../outputs/v2-050/nms_kitti2015_000000_10_range_048_warm/timing.json)

## TensorRT 转换审计

TensorRT 是单独对照。原生 PyTorch 5 FPS 验收只使用前面的两轮原生回放。

| 环节 | 实测结论 |
| --- | --- |
| Python 端两次 `.float()` | 三个实测引擎的输出绑定都是 FP32。两次调用都返回原张量，各约 0.002 ms，没有实际类型转换或张量拷贝。 |
| 引擎输入缓冲 | 输入是 FP32，缓冲拷贝约 1.6–2.1 ms。直绑实验的 `direct_calls=0`，预处理张量未满足连续 NCHW 条件，实际走了缓冲路径。直绑收益仍待用兼容输入复测。 |
| 引擎内 `reformat` | 三个尺寸的独立分层测量为 11.6–14.2 ms，记录自输入已在 GPU 上的 TensorRT 层级 profiler。 |

TensorRT + `bmm` 使用独立 CUDA 流时完成 599/600 帧，含排队 P95 为 1.66 秒。该组作为运行时对照保留。

## 代码与验证状态

- `scripts/benchmark_edge_stream.py` 的原生选项包括 `--precision amp-fp16`、`--precision encoder-fp16`、`--cudnn-tf32` 和 `--idisp-cost-backend bmm`。仅编码器 FP16 会在检测头之前把双目特征转回 FP32。
- 四帧诊断验证了运行和输出路径。原 40 帧冻结清单和带核验双目框的独立近距集仍待补齐，正式近车召回、去重与 P2 评估待完成。固定框工具已用公开帧的投影右框跑通 3 帧、21 个实例。
- 发布前运行了 46 项相关回归测试，全部通过。
- 原生优化回放的逐帧记录源自 `20261126-native-5hz/native-replay-5hz.json` 和 `native-layout-replay-5hz.json`。TF32、端到端和 TensorRT 分项分别源自 `20261126-followup` 与 `20261126-graph-combo` 的本地验证记录。

## 原生回放复现

在仓库根目录运行随报告保存的脚本。第一组原生对照使用 `yolo26-native-5hz.py`。布局短测生成 `native-layout-comparison.json`，随后布局回放脚本用该记录选择四帧平均处理时间最低的配置。 [原生对照脚本](../outputs/v2-050/idisp-fpn/jetson/20261126-native-5hz/yolo26-native-5hz.py) · [布局短测脚本](../outputs/v2-050/idisp-fpn/jetson/20261126-native-5hz/yolo26-native-layout.py) · [布局回放脚本](../outputs/v2-050/idisp-fpn/jetson/20261126-native-5hz/yolo26-native-layout-replay.py)

脚本从仓库读取四帧的标定记录。准备 KITTI 2015 双目图，从 `testing/image_2` 和 `testing/image_3` 取 `000000_10`，从 `training/image_2` 和 `training/image_3` 取其余三帧。分别放到 `/tmp/yolo26-mixed-frames/<帧号>_image_2.png` 和 `/tmp/yolo26-mixed-frames/<帧号>_image_3.png`。OpenStereo 的 `23d71c9` 源码位于 `/tmp/yolo26-openstereo`，Python 环境需要 timm 0.4.12。

```bash
export PYTHONPATH=src:scripts
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
run_dir=outputs/v2-050/idisp-fpn/jetson/20261126-native-5hz
python "$run_dir/yolo26-native-5hz.py"
python "$run_dir/yolo26-native-layout.py"
python "$run_dir/yolo26-native-layout-replay.py"
```
