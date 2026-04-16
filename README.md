# FOD Detection System (mini-target-detection)

## 项目简介
这是一个基于自监督学习的动态背景微小目标异常检测系统，专注于 Foreign Object Debris (FOD) 检测，例如跑道或道路上的异物。该系统采用教师-学生网络架构，通过学习“正常背景”来识别“异常物体”，无需大量手动标注的异常数据。

## 核心功能
- **自监督训练**：通过 Copy-Paste 策略自动生成训练数据，极大降低数据标注成本。
- **动态背景适应**：结合运动补偿（ego-motion）、时序滤波和注意力机制，有效处理视频中相机移动和光照变化。
- **高精度检测**：利用 CBAM（通道和空间注意力）和 ASPP（空洞空间金字塔池化）增强网络对微小目标和复杂背景的识别能力。
- **实时推理**：多线程异步处理视频流，实现高效检测。
- **GrabCut 精细化**：对检测到的目标进行像素级边缘优化。
- **ONNX 导出**：支持模型导出为 ONNX 格式，便于部署。

## 算法流程概述
1.  **训练阶段**：
    *   **教师-学生网络**：预训练的教师网络（ResNet18）提取正常背景特征，学生网络学习模拟教师网络的特征输出。
    *   **自监督数据生成**：将随机缩放和旋转的异物模板通过 Alpha 混合粘贴到正常背景图片上，生成带有“伪标签”的训练数据。
    *   **损失函数**：结合特征蒸馏损失（SmoothL1 + 余弦相似度）和分割损失（Focal Loss + Dice Loss），引导学生网络在异物区域产生高残差，并准确分割异物。
2.  **推理阶段**：
    *   **特征残差计算**：输入帧同时经过教师和学生网络，计算其特征差异。
    *   **注意力增强**：CBAM 和 ASPP 模块处理特征，提升对异物的敏感度并抑制背景噪声。
    *   **时序平滑**：采用 EMA 对连续帧的异常概率图进行平滑，减少闪烁。
    *   **后处理**：形态学操作（开运算、闭运算）去除噪声和连接断裂区域。
    *   **目标追踪**：利用 CentroidTracker 进行跨帧目标追踪，过滤瞬时噪声，确认持续出现的异常。
    *   **GrabCut优化**：对追踪到的目标区域进行 GrabCut 算法精细分割，获得更准确的异物轮廓。

## 安装指南

### 1. 克隆仓库
```bash
git clone https://github.com/hez8/mini-target-detection.git
cd mini-target-detection
```

### 2. 创建并激活虚拟环境 (可选但推荐)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 数据准备
*   **训练帧**：从正常背景视频中抽帧，放置到 `data/train_frames/` 目录下。
*   **异物模板**：准备带有透明背景（PNG，包含 Alpha 通道）的异物小图，放置到 `data/anomaly_templates/` 目录下。
*   **预处理模板**：运行 `prepare_templates.py` 来生成旋转后的模板（此步骤很重要，用于自监督训练）。
    ```bash
    python prepare_templates.py
    ```

## 使用方法

项目通过 `main.py` 作为统一入口，支持四种运行模式：

### 1. 训练模式
用于学习正常背景特征，并训练学生网络和分割头。
```bash
python main.py --mode train
```
训练过程中，模型权重会保存到 `checkpoints/` 目录下，TensorBoard 日志保存到 `logs/`。

### 2. 推理模式
对实时视频流或本地视频文件进行异物检测。
*   **使用摄像头 (设备 ID 为 0)**：
    ```bash
    python main.py --mode infer --video_source 0 --weights auto
    ```
*   **处理本地视频文件**：
    ```bash
    python main.py --mode infer --video_source data/raw_videos/road.mp4 --weights auto --save_video
    ```
    *`--weights auto` 会自动查找 `checkpoints/` 目录下最新的模型权重。*
    *`--save_video` 会将处理结果保存到 `data/test_videos_result/`。*
*   **指定模型权重**：
    ```bash
    python main.py --mode infer --video_source data/raw_videos/road.mp4 --weights checkpoints/YOUR_MODEL_DIR/best_student.pth
    ```

### 3. 热力图诊断模式
可视化网络输出的原始异常概率图（热力图），方便调试阈值和理解网络判断。
```bash
python main.py --mode map --video_source data/raw_videos/road.mp4 --weights auto --save_video
```
红色区域表示网络判定的高风险异常区域，蓝色为安全区域。

### 4. ONNX 导出模式
将训练好的 PyTorch 模型导出为 ONNX 格式，便于在其他框架或硬件上部署。
```bash
python main.py --mode export --weights auto
```
导出的模型文件将命名为 `anomaly_detector.onnx`。

## 配置
所有核心参数都可以在 `configs/default_config.py` 中进行调整，例如：
- `DEVICE`：`cuda` 或 `cpu`
- `INPUT_SIZE`：网络输入图片尺寸
- `ANOMALY_THRESHOLD`：异常二值化阈值
- `K_SIGMA`：Z-Score 动态阈值倍数（影响灵敏度）
- `TRACKER_MIN_HITS`：追踪器最少连续命中帧数（影响误报率）
- 训练参数：`BATCH_SIZE`, `LEARNING_RATE`, `EPOCHS` 等。
