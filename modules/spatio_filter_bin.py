# modules/spatio_filter.py
import cv2
import numpy as np
from modules.tracker import CentroidTracker
import os
import time

class SpatioTemporalFilter:
    def __init__(self, config):
        self.cfg = config
        # 读取配置，初始化目标时序跟踪器
        self.tracker = CentroidTracker(
            max_distance=self.cfg.TRACKER_MAX_DIST, 
            min_hits=self.cfg.TRACKER_MIN_HITS, 
            max_age=self.cfg.TRACKER_MAX_AGE
        )

    def extract_targets(self, full_anomaly_maps):
        """
        处理全分辨率异常图，提取边框，并经过 Tracker 洗礼
        :param full_anomaly_maps: 包含了多帧全尺寸异常图，只取最新一帧(中心帧)进行处理
        :return: 经过确认的 (Confirmed) 真实目标列表
        """
        # 取出最新构建好的 1080P/4K 热力图
        current_map = full_anomaly_maps[-1] if isinstance(full_anomaly_maps, list) else full_anomaly_maps
        
        # 将 float32 的热力图归一化到 0-255，转为 uint8 供 OpenCV 处理
        current_map_norm = cv2.normalize(current_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # test 热力图
        # # 1. 使用 Jet 色带将单通道灰度图转为伪彩色 (越红异常值越高)
        # heatmap_vis = cv2.applyColorMap(current_map_norm, cv2.COLORMAP_JET)
        
        # # 2. 自动创建用于存放调试图像的文件夹
        # debug_dir = "debug_heatmaps"
        # os.makedirs(debug_dir, exist_ok=True)
        
        # # 3. 使用精确到毫秒的时间戳命名，防止文件被覆盖
        # # (真实画面和热力图是严格同步的，你可以对比着看)
        # timestamp = int(time.time() * 1000)
        # save_path = os.path.join(debug_dir, f"heatmap_{timestamp}.jpg")
        
        # # 4. 写入磁盘 (不再使用 cv2.imshow 和 cv2.waitKey)
        # cv2.imwrite(save_path, heatmap_vis)
        # # =======================================================


        # 检测时直接将边缘区域的异常响应强行清零
        margin = self.cfg.EDGE_IGNORE_MARGIN
        if margin > 0:
            h, w = current_map_norm.shape
            # 安全校验
            if margin < h // 2 and margin < w // 2:
                current_map_norm[:margin, :] = 0        # 顶部边缘清零
                current_map_norm[h-margin:, :] = 0      # 底部边缘清零
                current_map_norm[:, :margin] = 0        # 左侧边缘清零
                current_map_norm[:, w-margin:] = 0      # 右侧边缘清零
        
        # 1. 自适应/固定阈值二值化
        thresh_val = int(self.cfg.ANOMALY_THRESHOLD * 255)
        _, binary_mask = cv2.threshold(current_map_norm, thresh_val, 255, cv2.THRESH_BINARY)
        
        # 2. 轻量级形态学开运算 (仅去除 1x1 的绝对噪点)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # 3. 连通域提取，构建本帧的 "疑似目标"
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
        
        raw_detections = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # 粗筛：仅仅过滤掉太大和太小的连通域
            if self.cfg.MIN_TARGET_AREA <= area <= self.cfg.MAX_TARGET_AREA:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                cx, cy = centroids[i]
                
                raw_detections.append({
                    "bbox": [x, y, w, h],
                    "area": area,
                    "centroid": (int(cx), int(cy))
                })
                
        # 4. [核心防误报防线] 送入跟踪器进行跨帧校验
        confirmed_targets = self.tracker.update(raw_detections)
        
        return confirmed_targets