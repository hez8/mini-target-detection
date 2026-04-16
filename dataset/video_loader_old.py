# dataset/video_loader.py
import os
import cv2
import glob
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import transforms
from configs.default_config import Config

class SelfSupervisedAnomalyDataset(Dataset):
    """
    自监督伪异常生成数据集。
    职责：加载正常帧 -> 避开边缘 -> 随机裁剪 -> 50%概率生成伪异常 -> 输出 (图像张量, Mask标签)
    """
    def __init__(self, frame_dir, transform=None):
        self.frame_dir = frame_dir
        self.patch_size = Config.INPUT_SIZE
        self.margin = getattr(Config, 'EDGE_IGNORE_MARGIN', 50)
        
        # 收集所有预处理好的图片路径
        self.frame_paths = self._extract_frame_paths()
        
        # 如果未传入 transforms，则使用标准的 ImageNet 归一化
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _extract_frame_paths(self):
        paths = glob.glob(os.path.join(self.frame_dir, "*.jpg")) + \
                glob.glob(os.path.join(self.frame_dir, "*.png"))
        if not paths:
            print(f"[警告] 目录 {self.frame_dir} 中没有找到任何图片！请先运行抽帧脚本。")
        return paths

    def __len__(self):
        return len(self.frame_paths)

    def generate_pseudo_anomaly(self, image_patch):
        """
        在 256x256 的图像块上，随机生成一块伪异常，并返回对应的 Mask 标签。
        """
        h, w = image_patch.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        
        # 50% 的概率保持为“纯正常背景”，让网络学习什么是安全的
        if random.random() < 0.5:
            return image_patch, mask
            
        # -----------------------------------------------------
        # 剩下 50% 概率：随机生成大小、位置不同的异常区块
        # -----------------------------------------------------
        # 随机宽高 (例如 10 到 40 像素，模拟石子、U盘等微小异物)
        bw = random.randint(10, 40)
        bh = random.randint(10, 40)
        
        # 随机位置
        bx = random.randint(0, w - bw - 1)
        by = random.randint(0, h - bh - 1)
        
        # 提取目标 ROI
        roi = image_patch[by:by+bh, bx:bx+bw]
        
        # 生成方法 1：彩色随机噪点 (破坏原有纹理，模拟杂物)
        noise = np.random.randint(0, 255, roi.shape, dtype=np.uint8)
        
        # 将噪点与原背景按 50% 透明度混合，模拟带环境反光的物体
        image_patch[by:by+bh, bx:bx+bw] = cv2.addWeighted(roi, 0.5, noise, 0.5, 0)
        
        # 也可以加入纯黑/纯白块来模拟高反光或深色异物 (取消注释即可激活)
        # if random.random() < 0.3:
        #     image_patch[by:by+bh, bx:bx+bw] = 0  # 纯黑块
        
        # 在 Mask 上记录该区域为真实异常 (1.0)
        mask[by:by+bh, bx:bx+bw] = 1.0 
        
        return image_patch, mask

    def __getitem__(self, idx):
        img_path = self.frame_paths[idx]
        # 读取图像并转为 RGB
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # ==========================================
        # 1. 边缘裁剪 (避开云台转动带来的黑边)
        # ==========================================
        h, w = image.shape[:2]
        if self.margin > 0 and self.margin < h // 2 and self.margin < w // 2:
            image = image[self.margin:h-self.margin, self.margin:w-self.margin]
            
        # ==========================================
        # 2. 随机裁剪为网络输入尺寸 (如 256x256)
        # ==========================================
        h_cropped, w_cropped = image.shape[:2]
        ph, pw = self.patch_size
        
        if h_cropped >= ph and w_cropped >= pw:
            # 随机起点
            top = random.randint(0, h_cropped - ph)
            left = random.randint(0, w_cropped - pw)
            patch = image[top:top+ph, left:left+pw]
        else:
            # 异常保护：如果原图实在太小，直接强行 resize (极少发生)
            patch = cv2.resize(image, (pw, ph))

        # ==========================================
        # 3. 在裁剪好的 Patch 上生成自监督伪异常
        # ==========================================
        # 为什么要在裁剪后生成？确保网络 100% 能看到这个微小的假目标
        aug_patch, gt_mask = self.generate_pseudo_anomaly(patch.copy())

        # ==========================================
        # 4. 数据格式转换 (转 Tensor)
        # ==========================================
        # 图像：经过 toTensor 和 归一化，变为 [3, H, W] 的 FloatTensor
        image_tensor = self.transform(aug_patch)
        
        # Mask：增加一个通道维度，变为 [1, H, W] 的 FloatTensor
        gt_mask_tensor = torch.from_numpy(gt_mask).unsqueeze(0)
        
        # 返回用于算蒸馏 Loss 的图像，以及用于算 BCE 分割 Loss 的标签图
        return image_tensor, gt_mask_tensor

# -------------------------------------------------------------------
# 兼容性封装 (为了不修改 train.py 的导入逻辑)
# 我们将旧的 NormalBackgroundDataset 指向新的 SelfSupervisedAnomalyDataset
# -------------------------------------------------------------------
NormalBackgroundDataset = SelfSupervisedAnomalyDataset