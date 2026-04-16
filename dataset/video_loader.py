# dataset/video_loader.py
import os
import cv2

# [核心优化] 强制关闭 OpenCV 内部多线程，彻底解除死锁
cv2.setNumThreads(0) 
cv2.ocl.setUseOpenCL(False)

import glob
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import transforms
from configs.default_config import Config

class SelfSupervisedAnomalyDataset(Dataset):
    """
    极速版自监督数据集：离线空间换时间 + 困难负样本挖掘 + GT 标签物理膨胀
    """
    def __init__(self, frame_dir, transform=None):
        self.frame_dir = frame_dir
        self.patch_size = getattr(Config, 'INPUT_SIZE', (256, 256))
        self.margin = getattr(Config, 'EDGE_IGNORE_MARGIN', 50)
        
        # 指向提前预旋转好的图库
        self.anomaly_template_dir = 'data/anomaly_templates_rotated'
        
        self.frame_paths = self._extract_frame_paths()
        self.template_paths = glob.glob(os.path.join(self.anomaly_template_dir, "*.png"))
        
        if not self.template_paths:
            print(f"[警告] 找不到预旋转图库：{self.anomaly_template_dir}，请确保已运行 prepare_templates.py！")
        
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _extract_frame_paths(self):
        paths = glob.glob(os.path.join(self.frame_dir, "*.jpg")) + \
                glob.glob(os.path.join(self.frame_dir, "*.png"))
        return paths

    def copy_paste_anomaly(self, bg_patch):
        h_bg, w_bg = bg_patch.shape[:2]
        mask = np.zeros((h_bg, w_bg), dtype=np.float32)
        
        # =========================================================
        # 50% 概率：不贴异物，引入“困难负样本”环境干扰
        # =========================================================
        if random.random() < 0.5 or not self.template_paths:
            # 20% 概率制造局部噪点、光斑、阴影
            if random.random() < 0.2:
                h_n, w_n = random.randint(10, 50), random.randint(10, 50)
                if h_bg > h_n and w_bg > w_n:
                    top_n = random.randint(0, h_bg - h_n)
                    left_n = random.randint(0, w_bg - w_n)
                    roi = bg_patch[top_n:top_n+h_n, left_n:left_n+w_n]
                    
                    if random.random() < 0.5:
                        noise = np.random.normal(0, 25, roi.shape).astype(np.float32)
                        bg_patch[top_n:top_n+h_n, left_n:left_n+w_n] = np.clip(roi + noise, 0, 255).astype(np.uint8)
                    else:
                        roi_blurred = cv2.GaussianBlur(roi, (15, 15), 0)
                        bg_patch[top_n:top_n+h_n, left_n:left_n+w_n] = np.clip(roi_blurred * random.uniform(0.5, 0.8), 0, 255).astype(np.uint8)
                        
            return bg_patch, mask
            
        # =========================================================
        # 50% 概率：极速 Copy-Paste 异物贴图
        # =========================================================
        template_path = random.choice(self.template_paths)
        template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
        
        if template is None or template.shape[2] != 4:
            return bg_patch, mask
        
        # 随机缩放
        target_size = random.randint(15, 60)
        h_t, w_t = template.shape[:2]
        scale = target_size / max(h_t, w_t)
        new_w, new_h = int(w_t * scale), int(h_t * scale)
        
        if new_w <= 2 or new_h <= 2:
            return bg_patch, mask
            
        template = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        obj_bgr = template[:, :, :3]
        obj_rgb = cv2.cvtColor(obj_bgr, cv2.COLOR_BGR2RGB)
        obj_alpha = template[:, :, 3] / 255.0 
        
        # 亮度抖动
        brightness_factor = random.uniform(0.5, 1.2)
        obj_rgb = np.clip(obj_rgb * brightness_factor, 0, 255).astype(np.uint8)

        if h_bg - new_h <= 0 or w_bg - new_w <= 0:
            return bg_patch, mask
            
        top = random.randint(0, h_bg - new_h - 1)
        left = random.randint(0, w_bg - new_w - 1)
        
        # Alpha 混合
        roi = bg_patch[top:top+new_h, left:left+new_w]
        alpha_3d = np.expand_dims(obj_alpha, axis=2)
        blended_roi = (obj_rgb * alpha_3d + roi * (1 - alpha_3d)).astype(np.uint8)
        bg_patch[top:top+new_h, left:left+new_w] = blended_roi
        
        # =========================================================
        # [核心优化] GT Mask 物理膨胀：强迫网络不漏边缘
        # =========================================================
        # 将 Alpha 阈值降到 0.15 捕获边缘阴影
        binary_mask = (obj_alpha > 0.15).astype(np.float32)
        
        # 5x5 内核进行形态学膨胀，向外扩充像素
        kernel = np.ones((5, 5), np.uint8)
        binary_mask_dilated = cv2.dilate(binary_mask, kernel, iterations=1)
        
        mask[top:top+new_h, left:left+new_w] = binary_mask_dilated

        return bg_patch, mask

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        img_path = self.frame_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w = image.shape[:2]
        if self.margin > 0 and self.margin < h // 2 and self.margin < w // 2:
            image = image[self.margin:h-self.margin, self.margin:w-self.margin]
            
        # 【尺度降维补丁】：全局缩小一半，避免原图大异物引发的视野塌陷
        image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

        h_cropped, w_cropped = image.shape[:2]
        ph, pw = self.patch_size
        
        if h_cropped >= ph and w_cropped >= pw:
            top = random.randint(0, h_cropped - ph)
            left = random.randint(0, w_cropped - pw)
            patch = image[top:top+ph, left:left+pw]
        else:
            patch = cv2.resize(image, (pw, ph))

        # 调用极速版造假
        aug_patch, gt_mask = self.copy_paste_anomaly(patch.copy())

        image_tensor = self.transform(aug_patch)
        gt_mask_tensor = torch.from_numpy(gt_mask).unsqueeze(0)
        
        return image_tensor, gt_mask_tensor

NormalBackgroundDataset = SelfSupervisedAnomalyDataset