# modules/spatio_filter.py
import cv2
import numpy as np
from modules.tracker import CentroidTracker

class SpatioTemporalFilter:
    def __init__(self, config):
        self.cfg = config
        # 初始化目标时序跟踪器
        self.tracker = CentroidTracker(
            max_distance=self.cfg.TRACKER_MAX_DIST, 
            min_hits=self.cfg.TRACKER_MIN_HITS, 
            max_age=self.cfg.TRACKER_MAX_AGE
        )

    def extract_targets(self, full_anomaly_maps):
        # 取出最新构建好的 float32 热力图
        current_map = full_anomaly_maps[-1] if isinstance(full_anomaly_maps, list) else full_anomaly_maps
        
        # =======================================================
        # 0. 边缘掩蔽 (Margin Masking)：强制忽略画面四周的无效区域
        # =======================================================
        # 使用 getattr 防止配置项未写引发报错，默认值为 0 (不裁剪)
        margin = getattr(self.cfg, 'EDGE_IGNORE_MARGIN', 0) 
        if margin > 0:
            h, w = current_map.shape
            # 安全校验：防止 margin 太大把整张图抹黑
            if margin < h // 2 and margin < w // 2:
                current_map[:margin, :] = 0        # 顶部边缘清零
                current_map[h-margin:, :] = 0      # 底部边缘清零
                current_map[:, :margin] = 0        # 左侧边缘清零
                current_map[:, w-margin:] = 0      # 右侧边缘清零

        # =======================================================
        # 1. 连贯性优化：高斯平滑 (消除尖锐噪点，让目标内部的响应融合)
        # =======================================================
        k_size = getattr(self.cfg, 'SMOOTH_KERNEL', 5)
        smoothed_map = cv2.GaussianBlur(current_map, (k_size, k_size), 0)

        # =======================================================
        # 2. 稳定性优化：Z-Score 动态自适应阈值计算
        # =======================================================
        # 排除绝对的 0 值区域（包含边缘黑边和刚才强制掩蔽的 margin 区域）
        # 这样算出来的均值和方差才是最纯粹的“有效背景”分布
        valid_pixels = smoothed_map[smoothed_map > 0]
        if len(valid_pixels) == 0:
            return [] # 全黑画面，无目标
            
        mu = np.mean(valid_pixels)
        sigma = np.std(valid_pixels)
        
        # 动态阈值 = 均值 + K 倍标准差
        k_sigma = getattr(self.cfg, 'K_SIGMA', 4.0)
        dynamic_thresh = mu + k_sigma * sigma
        
        # 直接在 float 级别进行二值化，将大于阈值的判定为 255 (异常)
        binary_mask = np.where(smoothed_map > dynamic_thresh, 255, 0).astype(np.uint8)

        # =======================================================
        # 3. 形态学修补：先闭运算(粘合目标)，再开运算(去小噪点)
        # =======================================================
        # 闭运算：桥接断开的区域，解决一个目标变多个框的问题
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # 开运算：去除离散的极小噪点
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel_open)
        
        # =======================================================
        # 4. 连通域提取与时序过滤
        # =======================================================
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
        
        raw_detections = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # 粗筛：过滤太小或太大的噪点/大面积误报
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
                
        # 5. 送入追踪器进行时序防闪烁校验
        confirmed_targets = self.tracker.update(raw_detections)
        
        return confirmed_targets