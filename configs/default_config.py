# configs/default_config.py
import torch
import os

class Config:
    # ---------------- 基础硬件配置 ----------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ---------------- 数据与路径配置 ----------------
    # 无监督训练用到的正常背景图片文件夹 (请提前用FFmpeg将正常视频抽帧放入此目录)
    TRAIN_DATA_DIR = os.path.join("data", "train_frames")
    
    # ---------------- 运动补偿配置 ----------------
    WINDOW_SIZE = 5            # 短时对齐窗口大小 (必须是奇数)
    
    # ---------------- 深度学习模型配置 ----------------
    # 师生网络骨干，可选: 'resnet18', 'mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_b0'
    BACKBONE = 'resnet18'      
    INPUT_SIZE = (256, 256)    # 网络输入的 Patch 尺寸 (必须是 32 的整数倍)
    
    # ---------------- 异常检测阈值与参数 ----------------
    ANOMALY_THRESHOLD = 0.6    # 异常图二值化阈值 (需根据实际数据的残差值灵活调整)
    KERNEL_OPEN = 2
    KERNEL_CLOSE = 3

    # Z-Score 动态阈值倍数。推荐值：3.0 ~ 5.0。
    # 值越大越严格(漏报增加)，值越小越敏感(误报增加)。
    K_SIGMA = 4.0 
    # [新增] 热力图高斯平滑核大小，必须是奇数。越大连贯性越好，但框会变大。
    SMOOTH_KERNEL = 7

    MIN_TARGET_DIAM = 5        # 目标最小像素直径 (滤除1x1孤立散粒噪点)
    MAX_TARGET_DIAM = 100      # 目标最大像素直径 (过滤大块的树叶反光误报)

    MIN_TARGET_AREA = 10        # 目标最小像素尺度 (滤除1x1孤立散粒噪点)
    MAX_TARGET_AREA = 10000      # 目标最大像素尺度 (过滤大块的树叶反光误报)

    EDGE_IGNORE_MARGIN = 20      # 训练和检测时忽略的边缘像素宽度
    
    # ---------------- Tracker 追踪器配置 ----------------
    TRACKER_MAX_DIST = 100      # 目标在相邻帧的最大位移像素 (防瞬移)
    TRACKER_MIN_HITS = 5       # 必须连续多少帧检测到才算真目标 (决定误报率)
    TRACKER_MAX_AGE = 10        # 丢失多少帧后彻底注销轨迹
    
    # ---------------- 训练参数 ----------------
    BATCH_SIZE = 16            # 如果显存不够，可降至 8
    NUM_WORKERS = 4
    PATIENCE = 50
    LEARNING_RATE = 1e-4
    EPOCHS = 1000