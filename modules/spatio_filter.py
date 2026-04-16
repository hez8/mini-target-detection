# modules/spatio_filter.py
import cv2
import numpy as np
from modules.tracker import CentroidTracker

class SpatioTemporalFilter:
    def __init__(self, config):
        self.cfg = config
        # Tracker 用于跨帧匹配目标，防止噪点一闪而过
        self.tracker = CentroidTracker(
            max_distance=getattr(self.cfg, 'TRACKER_MAX_DIST', 30), 
            min_hits=getattr(self.cfg, 'TRACKER_MIN_HITS', 3), 
            max_age=getattr(self.cfg, 'TRACKER_MAX_AGE', 2)
        )

    def extract_targets(self, current_mask):
        """
        接收端到端网络输出的 0/255 二值化 Mask 图 (uint8)，提取真实目标的 Bounding Box。
        """
        # 保险起见，强制转换为 OpenCV 支持的 uint8
        current_mask = current_mask.astype(np.uint8)

        # 直接提取连通域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(current_mask, connectivity=8)
        
        raw_detections = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # 使用 Config 中的面积限制进行粗筛 (过滤掉过小散斑或过大云雾)
            min_a = getattr(self.cfg, 'MIN_TARGET_AREA', 5)
            max_a = getattr(self.cfg, 'MAX_TARGET_AREA', 1000)
            
            if min_a <= area <= max_a:
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
                
        # 送入追踪器进行时序防闪烁过滤，只有连续出现的目标才会被输出
        confirmed_targets = self.tracker.update(raw_detections)
        
        return confirmed_targets