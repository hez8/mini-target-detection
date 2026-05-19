# Mini Target Detection System (mini-target-detection)

## 项目简介
这是一个基于自监督学习的微小目标异常检测系统。系统采用**教师-学生（Teacher-Student）蒸馏架构**，通过学习正常背景的特征分布来识别异常微小目标，支持多种轻量化骨干网络，兼顾检测精度与端侧推理速度。

## 核心功能
- **多骨干网络支持**：支持 ResNet18、**YOLOv11 (n/s)**、MobileNetV3 (Small/Large) 及 EfficientNet-B0，可通过命令行一键切换。
- **自监督训练**：通过 Copy-Paste 策略自动生成训练数据，无需手动标注异常样本。
- **增强型特征融合**：集成 **CBAM-Lite**（空间与通道注意力）与 **ASPP**（空洞空间金字塔池化），强化微小目标特征并压制环境干扰。
- **复合损失函数体系**：结合特征蒸馏损失（SmoothL1 + Cosine Similarity）与分割损失（Focal Loss + Dice Loss），提升边缘精度并抑制误报。
- **动态背景适应**：结合短时自运动补偿（Ego-motion）与时序平滑（EMA），有效处理光影变化。
- **智能追踪与优化**：内置 CentroidTracker 过滤瞬时噪声，并利用 GrabCut 算法进行像素级轮廓精细化。

## 算法流程
1.  **训练阶段**：
    *   **架构**：冻结预训练的教师网络，学生网络学习模拟其特征输出。对于 **YOLOv11** 骨干，系统会自动提取其特征提取阶段的 P1-P3 层特征图进行挂载与对齐。
    *   **数据**：将模板随机缩放、旋转并混合至正常背景，生成带有伪标签的 Patch。
    *   **优化**：在多尺度（H/2, H/4, H/8）特征上进行蒸馏，赋予网络极微小目标的捕捉能力。
2.  **推理阶段**：
    *   **残差计算**：通过师生网络特征差异生成初始异常图。
    *   **时序过滤**：利用追踪器确认持续出现的目标，排除随机闪烁的噪点。
    *   **后处理**：通过形态学操作与 GrabCut 获得最终的目标位置与边界。

==================================================================================================================
==================================================================================================================

## 完整操作流程

以下为从零开始的标准端到端工作流，涵盖数据准备、训练、到推理测试的完整链路。

### 第一步：数据采集与目录准备

将采集的道面视频按用途分类存放：

| 目录 | 用途 | 说明 |
|------|------|------|
| `data/train_videos/` | 训练集 | 纯背景、无 FOD 的正常道面视频 |
| `data/test_videos/` | 测试集 | 含有 FOD 目标的异常道面视频 |

```bash
# 示例：将本地单个视频上传至远程训练服务器
scp data/train_videos/*.mp4 user@172.19.47.11:8855/home/hz/fod_detect/data/train_videos/
scp data/test_videos/*.mp4  user@172.19.47.11:8855/home/hz/fod_detect/data/test_videos/

# 示例：将整个文件夹上传至远程服务器
scp -r data/train_videos/ user@172.19.47.11:8855/home/hz/fod_detect/data/train_videos/
scp -r data/test_videos/  user@172.19.47.11:8855/home/hz/fod_detect/data/test_videos/
```

### 第二步：视频抽帧

将 `data/train_videos/` 下的原始视频按固定帧率抽帧为训练图片，输出至 `data/train_frames/`。

```bash
python tools/data_preprocess.py
```

> **注意**：抽帧脚本的输入目录（`INPUT_DIR`）和输出目录（`OUTPUT_DIR`）定义在脚本内，可根据实际情况修改。默认输入为 `data/train_videos`，输出为 `data/train_frames`，后者需与配置文件中的 `TRAIN_DATA_DIR` 保持一致。

### 第三步：准备异常模板（可选）

若需训练 Copy-Paste 合成策略，准备带透明通道（RGBA）的 PNG 异物抠图放入 `data/anomaly_templates/`，然后运行旋转增强：

```bash
# 1. 创建模板目录并放入透明 PNG 图片
mkdir -p data/anomaly_templates

# 2. 生成旋转增强图库（每 10° 一张，共 36 个角度变体）
python tools/prepare_templates.py
```

生成结果将存入 `data/anomaly_templates_rotated/`，训练时自动加载。

### 第四步：训练模型

```bash
# 使用默认 ResNet18 骨干
python main.py --mode train

# 或指定其他骨干网络
python main.py --mode train --backbone yolo11n
```

模型权重将保存至 `checkpoints/<骨干名>_<时间戳>/best_student.pth`。

查看训练进度： tensorboard --logdir=logs --bind_all
登录http://172.19.47.11:6006/

### 第五步：模型测试

使用训练产出的模型对测试视频进行推理。`--weights auto` 会自动锁定与 `--backbone` 匹配的最新权重。

```bash
# 对单个视频进行推理并保存结果
python main.py --mode infer \
    --backbone yolo11n \
    --video_source data/test_videos/road_fod.mp4 \
    --weights auto \
    --save_video

# 对整个文件夹下的视频批量推理并保存结果
python main.py --mode infer \
    --backbone yolo11n \
    --video_source data/test_videos/ \
    --weights auto \
    --save_video

# 使用摄像头实时推理
python main.py --mode infer --backbone yolo11n --video_source 0 --weights auto
```

输出视频默认保存至 `results/infer/`；热力图诊断输出至 `results/map/`。

### 第六步：热力图诊断（可选）

当推理结果不理想时，使用热力图模式查看模型原始残差分布，辅助调整阈值：

```bash
python main.py --mode map \
    --backbone yolo11n \
    --video_source data/test_videos/road_fod.mp4 \
    --weights auto \
    --save_video
```
### 第七步： 导出模式（可选）
将 PyTorch 模型转换为 ONNX 格式，便于部署：
```bash
python main.py --mode export --backbone mobilenet_v3_large --weights auto
```
| `tools/export_torchscript.py` | 将模型导出为 TorchScript 格式（.pt），便于在无 Python 依赖的 C++ 环境中部署。支持 `--weights auto` 自动寻路。 | `python tools/export_torchscript.py --backbone yolo11n --weights auto --output detector.pt` |

### 关键路径速查

```
项目根目录
├── data/
│   ├── train_videos/              # 原始训练视频（纯背景）
│   ├── train_frames/              # 抽帧后的训练图片（预处理产出）
│   ├── anomaly_templates/         # 异常模板 PNG（手动放入）
│   ├── anomaly_templates_rotated/ # 旋转增强后的模板（脚本生成）
│   └── test_videos/               # 测试视频（含 FOD）
├── results/
│   ├── infer/                      # 推理输出视频
│   ├── map/                        # 热力图诊断输出
│   └── debug_dataset/              # 数据集调试样本
├── checkpoints/
│   └── <backbone>_<timestamp>/    # 训练权重
│       └── best_student.pth
├── configs/
│   └── default_config.py          # 全局配置
├── tools/                         # 辅助工具脚本
│   ├── data_preprocess.py         # 视频抽帧
│   ├── prepare_templates.py       # 模板旋转增强
│   ├── export_onnx.py             # ONNX 导出
│   ├── export_torchscript.py      # TorchScript 导出
│   ├── inference_map.py           # 热力图诊断
│   ├── inference_offline.py       # 离线推理
│   └── debug_*.py                 # 调试脚本
└── main.py                        # 统一入口
```

### 工具脚本详解

#### 模型导出

| 脚本 | 说明 | 使用示例 |
|------|------|----------|
| `tools/export_onnx.py` | 将训练好的 PyTorch 模型导出为 ONNX 格式，由 `main.py --mode export` 调用。支持动态 batch size，输出单通道异常热力图。 | `python main.py --mode export --backbone yolo11n --weights auto` |
| `tools/export_torchscript.py` | 将模型导出为 TorchScript 格式（.pt），便于在无 Python 依赖的 C++ 环境中部署。支持 `--weights auto` 自动寻路。 | `python tools/export_torchscript.py --backbone yolo11n --weights auto --output detector.pt` |

#### 推理与诊断

| 脚本 | 说明 | 使用示例 |
|------|------|----------|
| `tools/inference_map.py` | 热力图诊断模式，由 `main.py --mode map` 调用。输出网络的原始残差概率图（JET 伪彩色），用于直观判断异常阈值是否合理。 | `python main.py --mode map --video_source data/test_videos/ --weights auto --save_video` |
| `tools/inference_offline.py` | 基于 TorchScript 模型的离线推理工具，**无需完整项目依赖**，只需 PyTorch + OpenCV 即可运行。适合在边缘设备上快速部署验证。 | `python tools/inference_offline.py --model detector.pt --source data/test_videos/road.mp4 --thresh 0.45` |

#### 调试工具

| 脚本 | 说明 | 使用示例 |
|------|------|----------|
| `tools/debug_dataset.py` | 可视化训练数据集的质量。随机抽取 50 张经过 Copy-Paste 合成的训练样本，将原图、异常 Mask、叠加效果横向拼接保存至 `results/debug_dataset/`，用于检查数据增强效果。 | `python tools/debug_dataset.py` |
| `tools/debug_features.py` | 测试不同骨干网络（ResNet18、YOLO11n）的特征提取是否正常。打印各层输出张量的 Shape、Mean、Std、Max，用于验证新骨干接入后的计算图正确性。 | `python tools/debug_features.py` |

## 核心配置 (`configs/default_config.py`)
- `BACKBONE`: 默认骨干网络选择，可选: `'resnet18'`, `'yolo11n'`, `'yolo11s'`, `'mobilenet_v3_small'` 等。
- `INPUT_SIZE`: 网络输入的 Patch 尺寸（需为32的倍数）。
- `ANOMALY_THRESHOLD`: 异常图二值化阈值。
- `K_SIGMA`: Z-Score 动态阈值系数，用于自适应背景噪声。
- `TRACKER_MIN_HITS`: 追踪器最少确认帧数，增加此值可大幅降低误报。

## 依赖环境
- Python 3.8+
- PyTorch >= 1.8.0
- torchvision
- **ultralytics>=8.3.0** (用于 YOLOv11 支持)
- OpenCV (opencv-python)
- Tensorboard
