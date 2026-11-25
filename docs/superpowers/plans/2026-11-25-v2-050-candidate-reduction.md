# MobileNetV2 0.5× 候选减框实施方案

> 在 `experiment/mobilenetv2-050@cb779a1` 上实施。先固定第 24 轮权重验证推理阈值，再依据精度和速度结果决定是否训练检测分支。

**目标**　减少进入 iDisp 的低质量 Car 框对，保住第 24 轮已命中的近车与一物一对结果，验证半宽编码器节省的时间能否转化为端到端收益。

**权重**　0.5× 使用服务器正式训练第 24 轮的 `best_r2.pt`，SHA-256 为 `dd9990d5435d5da9696437009449bda4713d5d41a196a2d495e7363a3abd8095`。1.0× 对照使用仓库 `models/foreground_evaluation.pt`。沿用[半宽编码器实施计划](2026-11-25-mobilenetv2-050.md)的服务器与板端验收口径。

## 当前证据与修改优先级

| 40 帧 CPU 同帧对照 | 进入 iDisp 的框对 | 相对 1.0×/0.40 的 iDisp 对数差 | 10 米内输出框对 | 10 米内 Car 命中 |
| --- | ---: | ---: | ---: | ---: |
| 1.0×，score 0.40 | 50 | 0 | 26 | 18/23 |
| 0.5×，score 0.40 | 104 | +54 | 49 | 20/23 |
| 0.5×，score 0.45 | 83 | +33 | 41 | 20/23 |
| 0.5×，score 0.48 | **77** | **+27** | **38** | **20/23，同一批 GT** |
| 0.5×，score 0.50 | 71 | +21 | 35 | 19/23 |
| 0.5×，score 0.55 | 56 | +6 | 29 | 19/23 |
| 0.5×，score 0.40，粗筛改为 13 m | 88 | +38 | 46 | 19/23 |

分数阈值同为 0.40 时，0.5× 比 1.0× 多 54 对 iDisp 输入。其中 45 对是两目与同类 GT 的 IoU 都低于 0.50，5 对只有左目达到 0.50。该阈值下两版在已匹配的近车上各有 1 个额外重复框对。0.48 是本轮推荐进入独立复核的阈值。0.50 和 0.55 都漏掉 `000058_10:2`，13 米粗筛漏掉 `000047_10:1`。 [0.5× 逐框和阈值报告](../../../outputs/v2-050/box-diagnosis/v2-050.json) · [1.0× 逐框报告](../../../outputs/v2-050/box-diagnosis/v2.json)

## 任务 1　接通 0.48 的精度评估和回放

**改动**　[scripts/benchmark_edge_stream.py](../../../scripts/benchmark_edge_stream.py)、[scripts/evaluate_edge_near10.py](../../../scripts/evaluate_edge_near10.py)、[scripts/infer.py](../../../scripts/infer.py) 的现有模型加载入口。

- [ ] 给评估与回放入口加 `--score`，默认 `0.40`，显式传给 `load_published_model(..., score=...)`。单对推理继续使用现有 `--score`。给带 `--comparison-weights` 的评估加 `--comparison-score`，省略时沿用 `--score`。在报告写入实际阈值、权重 SHA、NMS/粗筛/去重参数。将 `0.48` 作为实验参数，待结果复核后再决定是否推广。
- [ ] 测试默认值、显式 `0.48`、非法阈值，以及同一个权重分别用 `0.48` 和 `0.40` 做交叉评估。确认两组采用各自的阈值，原始候选 ID 和真实 iDisp IDs 都可追踪。
- [ ] 固定全图、`legacy_clamp`、普通同类 paired NMS IoU 0.50、top-K 500、15 米粗筛、10 米精测、Mask 复核框 IoU 0.45、Mask IoU 0.50、`fixed_56` 和 33 个视差档。服务器先用相同精度复跑 1.0×/0.40、0.5×/0.40、0.5×/0.48。按 `frame_id + GT ID` 核对输出，原 20 辆近车都必须保住，确认同一对框通过 iDisp 和最终输出的数量达到 19/23。0.48 的一物一对数需本轮确认。
- [ ] 每组记录 NMS 前、NMS 后、粗筛后、iDisp 和最终输出框对数，以及近车 TP/FP/FN、Precision、Car Mask 前景有效覆盖、Pred-RoI EPE/D1。开发集以 **104 → 约 77 对、20/23 的目标身份完全一致** 为观察基准。记录低分候选被剔除的位置，并与同 checkpoint 的 0.40 做共同像素比较。
- [ ] 逐对报告保留 `frame_id + GT ID`、候选 ID、分数、iDisp 入口与最终输出。0.40 下的 `000018_10:2` 仍有 9584 与 9504 两对，分数分别为 0.689 和 0.535，都高于 0.48。现有 0.48 扫描只保存了目标命中和总框数，这一组列为重点复核，逐对重新记录。另汇总每个已匹配 GT 的额外框对数，即 `max(合格框对数 - 1, 0)`，iDisp 入口与最终输出分开计数。

## 任务 2　冻结参数后做独立样本与速度验证

- [ ] 固定 0.48，在与训练和当前 40 帧分开的近距样本上同帧对照 1.0×/0.40、0.5×/0.40 和 0.5×/0.48。逐类报告分母与实际命中，分母为 0 标 `N/A`。使用 `frame_id + GT ID` 比较 Car 的命中集合。**阈值推广的硬门槛是 0.5×/0.40 已命中的近车在 0.48 下零新增漏检。** 同时报告重复、Precision 和 iDisp 负载。对独立样本的任何阈值重调另作开发实验。

| 独立样本与类别 | 近距 GT 数 | 1.0×/0.40 命中 | 0.5×/0.40 命中 | 0.5×/0.48 命中 | 0.5×/0.40 已命中、0.48 丢失 |
| --- | ---: | ---: | ---: | ---: | --- |
| Car | 待测 | — | — | — | 数量与 `frame_id + GT ID` 列表 |
| Pedestrian | 待测 | — | — | — | 数量与 `frame_id + GT ID` 列表 |
| Truck | 待测 | — | — | — | 数量与 `frame_id + GT ID` 列表 |
| Bus | 待测 | — | — | — | 数量与 `frame_id + GT ID` 列表 |

- [ ] 对候选权重和阈值在同卡、相同推理精度和输入顺序下做 10／20 Hz 的 120 秒回放。三组均报告 Encoder、iDisp 和端到端 P95 的绝对值和同帧差值，另报完成数、超时、丢帧、积压和 iDisp 框对数 P50/P95。0.48 仍比 1.0× 多 27 对，速度按实测判断。用独立插桩轮计完整 iDisp 的逐帧 CUDA 耗时，无 iDisp 调用的帧记为 0，再从整组帧计算 P95。共享 GPU 结果单列。固定 RoI 对照可复用 [benchmark_edge_backbone.py](../../../scripts/benchmark_edge_backbone.py)，用来区分编码器收益与候选负载变化。
- [ ] 同一 0.5× 权重、阈值和后端一起通过精度与速度检查后，才作为部署候选。服务器 10 Hz 的基础目标为 1200/1200、P95 ≤100 ms、零丢帧且窗口结束无积压，20 Hz 的进阶目标为 2400/2400、P95 ≤50 ms。若 0.48 同时降低负载、保持精度并达到所需速度，本轮以阈值配置交付。

## 任务 3　阈值收益不足时微调 objectness

**触发条件**　0.48 在独立集新增近车漏检，或其 iDisp 负载仍使目标速度达不到。按任务 1 把候选拆成合格双目匹配、只有左目匹配、只有右目匹配、两目均未达 IoU 0.50、真实 10–15 米、真实 15 米外及无法确认距离。未匹配候选保留原有忽略标注口径。

- [ ] 从同一个第 24 轮完整半宽 checkpoint warm-start。新增明确的完整权重初始化路径，验证 0.5× 的 encoder、FPN、适配层和下游都正确回读，随后新建优化器。当前 0.5× `--init-weights` 走部分迁移，训练实验需要按完整状态启动。
- [ ] 做小规模 `floor` 与 `joint_iou` 对照。使用现有 `--one-to-one-score-target` 损失，两个实验只更新 `detector.one_to_one_levels.*.objectness`，冻结共享特征、类别分支、框回归、Mask 与 iDisp。冻结模块保持 eval 行为，回归测试核对更新前后的分类和 box6 张量一致。先试 3 个 epoch、学习率 `1e-4`、相同训练帧顺序。两组都沿用现有正负样本和 ignore mask 逻辑。
- [ ] `floor` 对正样本使用 `0.5 + 0.5 × 双目 IoU`，`joint_iou` 使用两目 IoU 的最小值作为目标。首轮比较同一阈值，再按开发集的同召回点比较 Precision 与 iDisp 对数。开发集先冻结阈值，再复用任务 2 的独立近距验证。 [目标分数损失](../../../src/lightbackbone_stereo_yolo/detection.py#L275)
- [ ] 两组用相同分箱报告分数筛选前的 objectness 概率直方图，按正样本、负样本和 ignore 分列，单列 0.40、0.48 附近的样本数量。对两目最小 IoU <0.50 的候选，报告 objectness 与最终检测分数的 P10/P50/P90、达到 0.48 的比例、NMS 后及 iDisp 入口数。合格近车候选报同样指标，便于判断两组分数是否分开。
- [ ] 3 轮后低 IoU 候选仍未减少时，先核对正／负／ignore 赋值数量、掩码覆盖、高分低 IoU 候选落入 ignore 的比例和进入 hard-negative 采样的数量。若分数诊断仍显示左／右框定位或配对错误是主要瓶颈，再另开框回归实验。保持宽度、输入和后处理口径，单独训练配对框回归支路，逐样本核对左右 IoU、右框中心误差及 10–15 米边界。后续宽度或 FPN 对照沿用原计划的独立实验口径。

## 第一轮执行命令

变量沿用原实施计划。评估和回放的 `--score`、`--comparison-score` 由任务 1 接入。

```bash
python scripts/evaluate_edge_near10.py \
  --openstereo-source "$STEREO_OPENSTEREO" --dataset-root "$STEREO_KITTI" \
  --split-json "$STEREO_VAL" --calibration-zip "$STEREO_CALIB" \
  --weights "$STEREO_V2_050_WEIGHTS" --comparison-weights "$STEREO_V2_050_WEIGHTS" \
  --score .48 --comparison-score .40 --device cuda:0 --precision fp32 \
  --near10-prefilter-m 15 --near10-review-box-iou .45 --near10-mask-iou .50 \
  --output "$STEREO_OUT/v2-050-score-048-vs-040.json"

for rate in 10 20; do
  python scripts/benchmark_edge_stream.py \
    --openstereo-source "$STEREO_OPENSTEREO" --dataset-root "$STEREO_KITTI" \
    --split-json "$STEREO_VAL" --calibration-zip "$STEREO_CALIB" \
    --weights "$STEREO_V2_050_WEIGHTS" --score .48 --device cuda:0 --precision fp32 \
    --near10-policy range-dedup --near10-prefilter-m 15 \
    --near10-review-box-iou .45 --near10-mask-iou .50 \
    --rate-fps "$rate" --duration-s 120 --output "$STEREO_OUT/v2-050-score-048-${rate}hz.json"
done
```
