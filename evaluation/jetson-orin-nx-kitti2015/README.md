# Jetson Orin NX 上的 KITTI Stereo 2015 单帧推理实测

2026 年 11 月 23 日，在 Jetson Orin NX 的 25 W 模式下，用发布的 Stereo-YOLO26 fixed_56 权重推理 KITTI Stereo 2015 测试集第 `000000_10` 帧。模型保留了 **50 组左右目匹配框**，预热后前向平均 **924.888 ms**。局部视差是耗时最大的分支，独立计时平均 **729.356 ms**。

## 硬件与软件

| 项目 | 实测环境 |
| --- | --- |
| 开发板 | NVIDIA Jetson Orin NX Engineering Reference Developer Kit |
| 模组与载板 | Jetson Orin NX 16 GB（P3767-0000），P3768-0000 载板，Tegra234 SoC |
| CPU | 8 核 Arm Cortex-A78AE，aarch64 |
| 功耗模式 | `nvpmodel` 25 W，模式 ID 3 |
| 系统 | Ubuntu 22.04.5 LTS，Linux 5.15.199-tegra |
| NVIDIA 软件栈 | JetPack 6.2.3，Jetson Linux 36.5.2，CUDA 12.6，cuDNN 9.3.0 |
| Python 环境 | Python 3.10.20，PyTorch 2.8.0，torchvision 0.23.0，timm 0.4.12，NumPy 1.26.4，Pillow 11.3.0 |
| 推理后端 | PyTorch，`cuda:0`，float32，`cudnn.deterministic=True`，`cudnn.benchmark=False` |

开发板名称和板号来自设备树，功耗模式来自 `nvpmodel -q`。25 W 是所选模式的标称值。[NVIDIA 模组与载板对照表](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/IN/QuickStart.html#jetson-modules-and-configurations)

## 输入与模型

左右图均为 1242×375 RGB。官方 ZIP 中的左目为 `testing/image_2/000000_10.png`，右目为 `testing/image_3/000000_10.png`。双向特征匹配得到 398 个垂直偏移不超过 1 像素的匹配点，其中 397 个满足 `x_left > x_right`，水平视差中位数为 22.59 像素。该检查用于确认左右顺序。[KITTI Stereo 2015](https://cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) · [官方数据 ZIP](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip)

| 输入 | SHA-256 |
| --- | --- |
| 左图 | `8b945d4c6a33924eae528b935414b12be24911a293c4033bf34f5b896407b521` |
| 右图 | `364f6b1b758c55014885a3dc9081408e889b7d8669c2ea6b64e37b270296ac60` |

代码基线为 `407e018f1d070b436203989e6d664fd87868899e`，OpenStereo 版本为 `23d71c92e33ad1f80dfc42bf29f5c6a914d38769`。推理使用发布权重 `stereo-yolo26-fixed56-seed17-e20.pt` 和 `scripts/infer.py` 的默认配置。One-to-one 开启并跳过 NMS，分数阈值 0.40，top-k 500，右框偏移为 `legacy_clamp`，RoI 采样固定为 56×56，局部视差使用 33 个残差档位。50 组匹配框均进入视差与掩码分支。[权重 SHA-256](../../models/SHA256SUMS) · [推理入口](../../scripts/infer.py)

从官方 ZIP 提取两张原图后，可以用以下命令复现推理输出。

```bash
python scripts/infer.py \
  --openstereo-source /path/to/OpenStereo \
  --left /path/to/testing/image_2/000000_10.png \
  --right /path/to/testing/image_3/000000_10.png \
  --device cuda:0 \
  --output outputs/kitti2015_000000_10-disparity.npz
```

## 计时结果

前向计时从图像张量已在 GPU 上开始，到 CUDA 同步完成结束。首次前向之后额外预热 3 次，然后测量 15 次前向。P95 使用线性插值。

| 前向统计 | 耗时（ms） |
| --- | ---: |
| 最小值 | 882.042 |
| 均值 | 924.888 |
| P50 | 898.530 |
| P95 | 1060.321 |
| 最大值 | 1150.488 |

冷启动记录使用同一进程的首次前向和首次输出，扣除插入的预热及重复计时。表中的写盘结束以压缩写入函数返回为准。

| 首次流程 | 耗时（ms） |
| --- | ---: |
| 初始化、加载权重和将模型移入 GPU | 770.897 |
| 读入两张图、归一化并上传 GPU | 83.916 |
| 首次模型前向 | 2209.650 |
| 结果转回 CPU 并整理 | 6.018 |
| 压缩写入 NPZ | 94.027 |
| 从推理入口到写盘 | 3164.574 |
| 从 Python 计时脚本入口到写盘，含库导入 | 9754.463 |

分阶段耗时来自另做的 10 次前向，使用 CUDA events。局部视差和掩码属于 RoI 整体的子项，阶段统计与上面的 15 次前向来自不同轮次。

| GPU 阶段 | 均值（ms） | P50（ms） | P95（ms） |
| --- | ---: | ---: | ---: |
| 共享特征编码器 | 196.555 | 190.037 | 230.922 |
| 检测头 | 20.274 | 20.632 | 23.825 |
| 检测解码与进入 RoI 前的准备 | 8.112 | 8.360 | 11.051 |
| RoI 整体 | 774.374 | 779.720 | 825.544 |
| 局部视差 | 729.356 | 735.393 | 780.940 |
| 掩码 | 21.421 | 21.511 | 24.964 |

仓库已有的 RTX 6000 Ada P95 为 250.039 ms，统计对象是 40 帧 KITTI 验证集，本报告的 Jetson P95 来自同一张测试帧的 15 次重复。两组数据可用于定位部署量级，硬件倍速需要在同一输入和软件配置下复测。[RTX 评估汇总](../summary.json)

## 推理产物

[下载 NPZ](000000_10-disparity.npz)。产物包含左右框、分数、类别、掩码概率、视差及其有效性、RoI 几何。视差数组形状为 `(50, 56, 56)`，单位是原图像素。程序标记有效的 156,800 个值均为有限数。该实验记录推理延迟与输出完整性，使用的是 KITTI 测试帧，未计算视差精度。

NPZ SHA-256 为 `65e016d34ba5761b7477fa8d6e7f9e236ab52b832653e41b5761a6d36b1825c9`。
