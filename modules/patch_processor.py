# modules/patch_processor.py
import numpy as np

class ImagePatcher:
    def __init__(self, patch_size=(256, 256), stride=(128, 128)):
        """
        初始化切图器
        :param patch_size: (height, width) 切片大小，需与网络输入一致
        :param stride: (stride_y, stride_x) 滑动步长。建议设为 patch_size 的一半，以保证 50% 的重叠率
        """
        self.patch_h, self.patch_w = patch_size
        self.stride_y, self.stride_x = stride

    def crop_to_patches(self, image: np.ndarray):
        """
        将大图切分为多个 Patch,并返回 Patch 列表及其对应的原图左上角坐标。
        :param image: shape 为 (H, W, C) 或 (H, W) 的 numpy 数组
        :return: patches_list, coords_list (格式为 [(x_min, y_min), ...])
        """
        img_h, img_w = image.shape[:2]
        patches = []
        coords = []

        # 计算切片的起始坐标点
        # 使用 min(y, img_h - patch_h) 确保最后一个 patch 严格贴合右下角边界，不越界也不补零
        y_starts = [min(y, img_h - self.patch_h) for y in range(0, img_h, self.stride_y)]
        x_starts = [min(x, img_w - self.patch_w) for x in range(0, img_w, self.stride_x)]

        # 去重（如果图像尺寸不能被步长整除，最后两个起点可能会重复）
        y_starts = sorted(list(set(y_starts)))
        x_starts = sorted(list(set(x_starts)))

        for y in y_starts:
            for x in x_starts:
                patch = image[y:y + self.patch_h, x:x + self.patch_w]
                patches.append(patch)
                coords.append((x, y))

        return patches, coords

    def stitch_anomaly_maps(self, patch_maps, coords, original_shape):
        """
        将网络输出的各个 Patch 的异常热力图，拼接回原图尺寸。
        对于重叠区域，采用平均值法平滑过渡，消除接缝线的误报。
        :param patch_maps: 异常图 Patch 列表，单个 shape 为 (patch_h, patch_w)
        :param coords: 对应的坐标列表 [(x_min, y_min), ...]
        :param original_shape: 原图尺寸 (H, W)
        :return: 拼接后的全尺寸异常热力图 (H, W)
        """
        target_h, target_w = original_shape[:2]
        
        # global_map 用于累加异常值
        global_map = np.zeros((target_h, target_w), dtype=np.float32)
        # weight_map 用于记录每个像素被多少个 patch 覆盖（即重叠次数）
        weight_map = np.zeros((target_h, target_w), dtype=np.float32)

        for patch_map, (x, y) in zip(patch_maps, coords):
            global_map[y:y + self.patch_h, x:x + self.patch_w] += patch_map
            weight_map[y:y + self.patch_h, x:x + self.patch_w] += 1.0

        # 防止除以 0（理论上全覆盖不会有 0）
        weight_map = np.maximum(weight_map, 1.0)
        
        # 计算加权平均，消除重叠边缘产生的伪影
        final_map = global_map / weight_map
        
        return final_map