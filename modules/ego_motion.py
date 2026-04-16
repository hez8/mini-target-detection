# modules/ego_motion.py
import cv2
import numpy as np
import logging

class EgoMotionCompensator:
    def __init__(self, window_size=5):
        """
        初始化短时自运动补偿器
        :param window_size: 对齐窗口大小 (必须为奇数)
        """
        assert window_size % 2 != 0, "Window size must be odd."
        self.window_size = window_size
        self.center_idx = window_size // 2
        
        # 使用 ORB 提取特征点，追求极致速度
        self.orb = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def _estimate_affine(self, src_img, dst_img):
        """
        计算从 src 到 dst 的刚体平移/旋转矩阵 (带有 ECC 保底机制)
        :param src_img: 待对齐的灰度图
        :param dst_img: 目标基准灰度图 (中心帧)
        :return: 2x3 的仿射变换矩阵
        """
        kp1, des1 = self.orb.detectAndCompute(src_img, None)
        kp2, des2 = self.orb.detectAndCompute(dst_img, None)
        
        matrix = None
        
        # ========================================================
        # 阶段 1: 尝试 ORB + RANSAC 特征匹配 (速度最快，适用于纹理丰富场景)
        # ========================================================
        if des1 is not None and des2 is not None and len(des1) >= 10 and len(des2) >= 10:
            matches = self.matcher.match(des1, des2)
            # 按汉明距离排序，保留最好的匹配点
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:50]
            
            if len(good_matches) >= 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                # 计算包含平移、旋转、缩放的仿射矩阵
                matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
        
        # ========================================================
        # 阶段 2: 如果 ORB 失败 (如遇到纯平纹理)，启动 ECC 梯度匹配保底
        # ========================================================
        if matrix is None:
            # 云台慢扫场景下，直接约束为纯平移模型 (MOTION_TRANSLATION) 以成倍提速
            warp_mode = cv2.MOTION_TRANSLATION 
            matrix = np.eye(2, 3, dtype=np.float32)
            
            # ECC 的迭代终止条件：迭代 50 次或精度达到 0.001
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
            
            try:
                # [核心提速技巧]：缩小分辨率计算 ECC，速度可提升 3-4 倍
                src_small = cv2.resize(src_img, (0, 0), fx=0.5, fy=0.5)
                dst_small = cv2.resize(dst_img, (0, 0), fx=0.5, fy=0.5)
                
                _, matrix = cv2.findTransformECC(src_small, dst_small, matrix, warp_mode, criteria)
                
                # 由于是在降采样一半的图上求的平移量，需要乘 2 还原回原分辨率
                matrix[0, 2] *= 2.0
                matrix[1, 2] *= 2.0
                
            except cv2.error:
                # ========================================================
                # 阶段 3: 极端恶劣工况 (图像全黑或严重撕裂)
                # ========================================================
                logging.debug("ORB 和 ECC 均失效，返回不运动矩阵。")
                matrix = np.eye(2, 3, dtype=np.float32)
                
        return matrix

    def align_buffer(self, frame_buffer):
        """
        将 Buffer 中所有的多帧图像对齐到中心帧坐标系
        :param frame_buffer: 包含多帧 BGR 图像的列表 (长度为 window_size)
        :return: 对齐后的 BGR 图像列表
        """
        center_frame = frame_buffer[self.center_idx]
        h, w = center_frame.shape[:2]
        aligned_frames = []
        
        # 提前将中心帧转为灰度图，避免在循环中重复计算
        center_gray = cv2.cvtColor(center_frame, cv2.COLOR_BGR2GRAY)
        
        for i, frame in enumerate(frame_buffer):
            # 中心帧自己不需要对齐
            if i == self.center_idx:
                aligned_frames.append(frame)
                continue
                
            # 计算灰度图用于提取位移参数，但实际 Warp 操作应用在彩色原图上
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            matrix = self._estimate_affine(frame_gray, center_gray)
            
            # cv2.INTER_LINEAR 双线性插值在速度和画质上最均衡
            aligned_frame = cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LINEAR)
            aligned_frames.append(aligned_frame)
            
        return aligned_frames