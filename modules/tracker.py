# modules/tracker.py
import numpy as np
from collections import OrderedDict

class Track:
    def __init__(self, target_id, bbox, centroid, area):
        self.id = target_id
        self.bbox = bbox              # [x, y, w, h]
        self.centroid = centroid      # (cx, cy)
        self.area = area              # 像素面积
        self.hits = 1                 # 连续命中(匹配)的次数
        self.time_since_update = 0    # 连续丢失的帧数
        self.is_confirmed = False     # 是否被确认为真实目标

class CentroidTracker:
    def __init__(self, max_distance=30, min_hits=3, max_age=2):
        """
        初始化质心跟踪器
        :param max_distance: 相邻帧之间允许的最大质心移动像素距离（过滤不可能的瞬移噪点）
        :param min_hits: 连续成功匹配多少帧后，才确认该目标为真（极值/噪点过滤的核心）
        :param max_age: 目标丢失多少帧后将其彻底注销
        """
        self.next_target_id = 0
        self.tracks = OrderedDict()
        
        self.max_distance = max_distance
        self.min_hits = min_hits
        self.max_age = max_age

    def register(self, bbox, centroid, area):
        """注册一个新发现的疑似目标"""
        self.tracks[self.next_target_id] = Track(self.next_target_id, bbox, centroid, area)
        self.next_target_id += 1

    def deregister(self, target_id):
        """注销一个已丢失的目标"""
        del self.tracks[target_id]

    def update(self, detections):
        """
        核心跟踪逻辑：将当前帧检测到的目标与历史轨迹进行匹配
        :param detections: 列表，包含当前帧所有疑似目标 [{"bbox": [x,y,w,h], "centroid": (cx,cy), "area": a}, ...]
        :return: 经过确认的 (Confirmed) 真实目标列表
        """
        # 1. 如果当前帧没有任何检测结果
        if len(detections) == 0:
            for target_id in list(self.tracks.keys()):
                self.tracks[target_id].time_since_update += 1
                if self.tracks[target_id].time_since_update > self.max_age:
                    self.deregister(target_id)
            return self._get_confirmed_tracks()

        # 2. 提取当前帧的所有质心坐标
        input_centroids = np.zeros((len(detections), 2), dtype=int)
        for i, det in enumerate(detections):
            input_centroids[i] = det['centroid']

        # 3. 如果当前没有任何历史轨迹，直接全部注册为新目标
        if len(self.tracks) == 0:
            for det in detections:
                self.register(det['bbox'], det['centroid'], det['area'])
            return self._get_confirmed_tracks()

        # 4. 获取历史轨迹的质心
        track_ids = list(self.tracks.keys())
        track_centroids = np.zeros((len(track_ids), 2), dtype=int)
        for i, track_id in enumerate(track_ids):
            track_centroids[i] = self.tracks[track_id].centroid

        # 5. 计算 历史质心 与 当前检测质心 的距离矩阵 (纯 numpy 实现，无需外部库)
        # 结果 D[i, j] 表示第 i 个历史目标与第 j 个当前检测目标的距离
        D = np.linalg.norm(track_centroids[:, np.newaxis] - input_centroids, axis=-1)

        # 6. 贪心匹配逻辑 (找到距离最近的配对)
        # 获取矩阵每一行的最小值的索引 (历史目标倾向于匹配哪个当前目标)
        rows = D.min(axis=1).argsort()
        # 获取矩阵每一列的最小值的索引 (当前目标倾向于被哪个历史目标匹配)
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            # 如果距离大于我们设定的最大运动阈值，说明是毫无关联的噪点跳跃，拒绝匹配
            if D[row, col] > self.max_distance:
                continue

            # 匹配成功，更新轨迹状态
            track_id = track_ids[row]
            det = detections[col]
            
            self.tracks[track_id].bbox = det['bbox']
            self.tracks[track_id].centroid = det['centroid']
            self.tracks[track_id].area = det['area']
            self.tracks[track_id].hits += 1               # 命中次数 +1
            self.tracks[track_id].time_since_update = 0   # 重置丢失计数
            
            # 状态机：达到阈值，确认为真实目标
            if self.tracks[track_id].hits >= self.min_hits:
                self.tracks[track_id].is_confirmed = True

            used_rows.add(row)
            used_cols.add(col)

        # 7. 处理未匹配上的历史目标 (丢失了)
        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        for row in unused_rows:
            track_id = track_ids[row]
            self.tracks[track_id].time_since_update += 1
            if self.tracks[track_id].time_since_update > self.max_age:
                self.deregister(track_id)

        # 8. 处理未匹配上的当前检测目标 (新出现的)
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)
        for col in unused_cols:
            det = detections[col]
            self.register(det['bbox'], det['centroid'], det['area'])

        return self._get_confirmed_tracks()

    def _get_confirmed_tracks(self):
        """仅返回已确认的真实目标，过滤掉所有昙花一现的噪点"""
        confirmed = []
        for track_id, track in self.tracks.items():
            if track.is_confirmed and track.time_since_update == 0:
                confirmed.append({
                    "id": track.id,
                    "bbox": track.bbox,
                    "centroid": track.centroid,
                    "area": track.area
                })
        return confirmed