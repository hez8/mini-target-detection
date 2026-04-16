# debug_dataset.py
import os
import cv2
import torch
import numpy as np
from configs.default_config import Config
from dataset.video_loader import NormalBackgroundDataset

def unnormalize(tensor):
    """将归一化后的 Tensor 还原为可视化的 numpy 图像 (RGB)"""
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    
    img = tensor.permute(1, 2, 0).numpy() # [H, W, 3]
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img

def main():
    cfg = Config()
    
    # 确保保存目录存在
    save_dir = "debug_dataset_outputs"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"[*] 正在加载数据集: {cfg.TRAIN_DATA_DIR}")
    dataset = NormalBackgroundDataset(frame_dir=cfg.TRAIN_DATA_DIR)
    
    if len(dataset) == 0:
        print("[-] 数据集为空！请检查路径。")
        return

    print(f"[*] 成功加载数据集。开始抽取 10 张样本...")
    
    count = 0
    # 随机打乱抽取
    indices = np.random.permutation(len(dataset))
    
    for idx in indices:
        if count >= 50:
            break
            
        img_tensor, mask_tensor = dataset[idx]
        
        # 1. 还原图像 (转回 BGR 供 OpenCV 保存)
        img_rgb = unnormalize(img_tensor)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # 2. 还原 Mask (把 0~1 的 FloatTensor 转为 0~255 的灰度图)
        mask = mask_tensor.squeeze(0).numpy()
        mask_vis = (mask * 255).astype(np.uint8)
        
        # 3. 画个红色的半透明叠加图，看看 Mask 贴合得好不好
        overlay = img_bgr.copy()
        overlay[mask_vis > 0] = (0, 0, 255) # BGR
        blended = cv2.addWeighted(overlay, 0.5, img_bgr, 0.5, 0)
        
        # 4. 将 原图、Mask、叠加图 横向拼接到一起
        mask_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((img_bgr, mask_bgr, blended))
        
        # 保存
        save_path = os.path.join(save_dir, f"sample_{count}.jpg")
        cv2.imwrite(save_path, combined)
        print(f"[+] 已保存测试图 -> {save_path}")
        
        # 只要生成的 Mask 中有异常物体，计数才加1 (因为有50%概率是纯背景)
        if mask.sum() > 0:
            count += 1

    print(f"[*] 请打开 {save_dir} 文件夹查看造假效果！")

if __name__ == "__main__":
    main()