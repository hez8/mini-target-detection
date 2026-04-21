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

## 使用方法

项目通过 `main.py` 作为统一入口，所有模式均支持 `--backbone` 参数。你也可以在 `configs/default_config.py` 中预设默认骨干。

### 1. 训练模式
使用指定的骨干网络启动无监督训练：
```bash
# 默认使用 resnet18
python main.py --mode train

# 使用轻量化 MobileNetV3 Large
python main.py --mode train --backbone mobilenet_v3_large

# 使用 YOLOv11n 骨干
python main.py --mode train --backbone yolo11n
```
*训练产生的权重将保存至 `checkpoints/骨干名_时间戳/` 目录下。*

### 2. 推理模式
对视频或摄像头流进行实时检测：
```bash
# 自动寻找最新训练的模型进行推理
python main.py --mode infer --video_source data/raw_videos/road.mp4 --backbone mobilenet_v3_large --weights auto --save_video

# 使用摄像头进行实时推理
python main.py --mode infer --video_source 0 --weights auto
```
*注意：推理时指定的 `--backbone` 必须与训练时一致。`--weights auto` 会自动锁定对应骨干下最新的权重文件。*

### 3. 热力图诊断模式
可视化网络输出的原始残差分布（热力图），用于调试阈值：
```bash
python main.py --mode map --video_source data/raw_videos/road.mp4 --backbone mobilenet_v3_large
```

### 4. ONNX 导出模式
将 PyTorch 模型转换为 ONNX 格式，便于部署：
```bash
python main.py --mode export --backbone mobilenet_v3_large --weights auto
```

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
