# train.py
import os
import time
import logging

# =================================================================
# [核心性能优化]：强制关闭 OpenCV 内部多线程，彻底解除与 PyTorch 的死锁争抢
# 必须写在 import torch 和其他耗时库之前！
# =================================================================
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

# 导入自定义模块
from configs.default_config import Config
from models.distillation_net import TeacherStudentNet
from dataset.video_loader import NormalBackgroundDataset

# ==========================================
# 工具类 1：复合特征提取损失 (Distillation Loss)
# ==========================================
class DistillationLoss(nn.Module):
    """
    结合 MSE (均方误差) 与 Cosine Similarity (余弦相似度) 的复合特征损失。
    MSE 负责约束特征的绝对幅值，Cosine 负责约束特征（纹理走向）的高维方向。
    """
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.cos = nn.CosineSimilarity(dim=1)
        self.smooth_l1 = nn.SmoothL1Loss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, t_feats, s_feats):
        total_loss = 0
        for t, s in zip(t_feats, s_feats):
            # 1. 计算均方误差
            l1_loss = self.smooth_l1(s, t)
            t_norm = F.normalize(t, p=2, dim=1)
            s_norm = F.normalize(s, p=2, dim=1)
            cos_loss = 1.0 - torch.mean(torch.sum(t_norm * s_norm, dim=1))
            # mse_loss = self.mse(s, t)
            # # 2. 计算余弦损失 (cos_sim 越接近1表示越相似，所以用 1 - cos_sim 作为惩罚)
            # cos_loss = torch.mean(1 - self.cos(s, t))
            
            total_loss += self.alpha * l1_loss + self.beta * cos_loss
        return total_loss

# ==========================================
# 工具类 2：Dice Loss (解决掩膜收缩与边缘置信度衰减)
# ==========================================
class DiceLoss(nn.Module):
    """
    通过全局交并比 (IoU) 来计算损失，强迫网络不仅要找到目标中心，
    还要将高置信度蔓延并覆盖完整的物理边缘。
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 展平张量
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 计算交集
        intersection = (inputs * targets).sum()                            
        # 计算 Dice 系数 (2 * 交集 / (预测面积 + 真实面积))
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)  
        
        # 返回 Loss (1 - Dice，越小越好)
        return 1 - dice

# ==========================================
# 工具类 3：Focal Loss (新增：专门用于压制高频噪点误报)
# ==========================================
class FocalLoss(nn.Module):
    """
    降低简单背景像素的权重，将计算火力集中在惩罚模棱两可的假阳性噪点上。
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        # reduction='none' 保留每个像素的 loss，以便后续乘上权重矩阵
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        # 巧妙利用 exp(-bce) 获取目标类别的预测概率 pt
        pt = torch.exp(-bce_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# ==========================================
# 工具类 4：早停机制 (Early Stopping)
# ==========================================
class EarlyStopping:
    """当验证集 Loss 不再下降时，提前停止训练，防止过拟合"""
    def __init__(self, patience=30, min_delta=1e-4): # patience 已放宽至 30
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            logging.info(f"EarlyStopping 计数: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# ==========================================
# 训练主引擎
# ==========================================
def setup_logger(log_dir):
    """配置日志记录器"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "training.log")),
            logging.StreamHandler()
        ]
    )

def train_pipeline():
    cfg = Config()
    
    # 1. 目录准备
    exp_time = time.strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join("checkpoints", exp_time)
    log_dir = os.path.join("logs", exp_time)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    setup_logger(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    logging.info(f"=== 启动 True E2E 极速版异常分割训练 ===")
    logging.info(f"使用设备: {cfg.DEVICE}")

    # 2. 数据集与 DataLoader 配置
    full_dataset = NormalBackgroundDataset(frame_dir=cfg.TRAIN_DATA_DIR)
    
    # 划分 90% 训练集，10% 验证集
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 【优化】：根据你的 CPU 核心数，将 num_workers 调大至 8 (Windows 下若报错请改回 0)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, 
                              num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, 
                            num_workers=8, pin_memory=True)
    
    logging.info(f"数据加载完成: 训练集 {train_size} 个 Patch, 验证集 {val_size} 个 Patch")

    # 3. 初始化端到端模型
    model = TeacherStudentNet().to(cfg.DEVICE)
    model.student.train() 
    model.segmenter.train()
    
    # 4. 损失函数体系配置
    criterion_distill = DistillationLoss(alpha=1.0, beta=0.5) 
    # 【修改】：用 Focal Loss 完全替换 BCE Loss
    criterion_seg_focal = FocalLoss(alpha=0.25, gamma=2.0)
    criterion_seg_dice = DiceLoss()     

    # 5. 优化器与调度器
    optimizer = AdamW([
        {'params': model.student.parameters()},
        {'params': model.segmenter.parameters()}
    ], lr=cfg.LEARNING_RATE, weight_decay=1e-4)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=cfg.PATIENCE) 

    # 6. 核心训练循环
    best_val_loss = float('inf')
    
    for epoch in range(cfg.EPOCHS):
        model.student.train()
        model.segmenter.train()
        
        train_loss_epoch = 0.0
        train_distill_epoch = 0.0
        train_seg_epoch = 0.0
        
        for batch_idx, (batch_imgs, gt_masks) in enumerate(train_loader):
            batch_imgs = batch_imgs.to(cfg.DEVICE, non_blocking=True)
            gt_masks = gt_masks.to(cfg.DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 前向传播 (一口气输出特征和 Mask)
            t_feats, s_feats, pred_masks = model(batch_imgs)
            
            # 损失 1：计算特征蒸馏 Loss
            # 【核心修改点：将 [:2] 改为 [:3]，涵盖 Layer0、Layer1、Layer2，赋予网络浅层特征的“嗅觉”】
            distill_loss = criterion_distill(t_feats[:3], s_feats[:3])
            
            # 损失 2：计算分割联合 Loss (Focal + Dice)
            loss_focal = criterion_seg_focal(pred_masks, gt_masks)
            loss_dice = criterion_seg_dice(pred_masks, gt_masks)
            seg_loss = loss_focal + loss_dice
            
            # 联合总 Loss，加重蒸馏权重
            loss = 2.0 * distill_loss + seg_loss
            
            # 反向传播与优化
            loss.backward()
            optimizer.step()
            
            train_loss_epoch += loss.item()
            train_distill_epoch += distill_loss.item()
            train_seg_epoch += seg_loss.item()
            
            # 打印 Step 日志
            if batch_idx % 10 == 0:
                logging.info(f"Epoch [{epoch+1}/{cfg.EPOCHS}] Step [{batch_idx}/{len(train_loader)}] "
                             f"Total: {loss.item():.4f} (Distill: {distill_loss.item():.4f}, Seg: {seg_loss.item():.4f})")
                
                step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Training/Total_Loss', loss.item(), step)
                writer.add_scalar('Training/Distill_Loss', distill_loss.item(), step)
                writer.add_scalar('Training/Seg_Loss_Total', seg_loss.item(), step)
                writer.add_scalar('Training/Focal_Loss', loss_focal.item(), step)
                writer.add_scalar('Training/Dice_Loss', loss_dice.item(), step)

        avg_train_loss = train_loss_epoch / len(train_loader)
        
        # 7. 验证环节 (Eval)
        model.student.eval()
        model.segmenter.eval()
        val_loss_epoch = 0.0
        
        with torch.no_grad():
            for batch_imgs, gt_masks in val_loader:
                batch_imgs = batch_imgs.to(cfg.DEVICE, non_blocking=True)
                gt_masks = gt_masks.to(cfg.DEVICE, non_blocking=True)
                
                t_feats, s_feats, pred_masks = model(batch_imgs)
                
                # 【核心修改点：验证集同样扩展至 [:3]】
                distill_loss = criterion_distill(t_feats[:3], s_feats[:3])
                
                loss_focal = criterion_seg_focal(pred_masks, gt_masks)
                loss_dice = criterion_seg_dice(pred_masks, gt_masks)
                seg_loss = loss_focal + loss_dice
                
                # 验证集也使用同样的联合权重
                val_loss = 2.0 * distill_loss + seg_loss
                val_loss_epoch += val_loss.item()
                
        avg_val_loss = val_loss_epoch / len(val_loader)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 写入 TensorBoard
        writer.add_scalar('Epoch_Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Epoch_Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('HyperParams/Learning_Rate', current_lr, epoch)
        
        logging.info(f"=== Epoch [{epoch+1}/{cfg.EPOCHS}] 总结 ===")
        logging.info(f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")
        
        # 8. Checkpoint 保存机制
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(ckpt_dir, "best_student.pth")
            
            # 保存整个 model 的 state_dict
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"[*] 发现更优模型，已完整保存至: {best_model_path}")
            
        # 常规保存 (每 5 个 Epoch)
        if (epoch + 1) % 5 == 0:
            periodic_path = os.path.join(ckpt_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), periodic_path)
            
        # 早停检查
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            logging.warning("触发早停机制，验证集 Loss 不再下降，训练提前结束！")
            break

    writer.close()
    logging.info("=== 真·端到端训练流水线执行完毕 ===")

if __name__ == "__main__":
    train_pipeline()