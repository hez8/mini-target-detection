# models/distillation_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# =========================================================
# 模块 1: 空间与通道注意力机制 (CBAM-Lite)
# 作用：在特征融合阶段，压制路面标线干扰，强化异物特征
# =========================================================
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAMBlock, self).__init__()
        # 通道注意力 (Channel Attention)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # 空间注意力 (Spatial Attention)
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # 1. 应用通道注意力
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid_channel(avg_out + max_out)
        x = x * out

        # 2. 应用空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        spatial_out = self.sigmoid_spatial(self.conv_spatial(spatial_out))
        
        return x

# =========================================================
# 【新增模块】：轻量级 ASPP (空洞空间金字塔)
# 作用：扩大感受野，让网络结合“大环境”判断局部边缘，免疫裂缝和砖块
# =========================================================
class LightweightASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LightweightASPP, self).__init__()
        # 1. 局部细节视野 (dilaion=1)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 2. 中等环境视野 (dilation=3) - 跨越水渍
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 3. 宏观全局视野 (dilation=6) - 识别路面结构
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 融合多尺度上下文
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return self.conv_out(out)

# =========================================================
# 模块 2: 增强版 ResNet 特征提取器
# 修改点：增加了对 Layer 0 (Stem) 的提取，保留 H/2 的超高分辨率
# =========================================================
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        
        # Layer 0 (Stem 层): H/2, W/2, 64通道 - 捕捉极微小特征的关键
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        # Layer 1: H/4, W/4, 64通道
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        # Layer 2: H/8, W/8, 128通道
        self.layer2 = resnet.layer2

    def forward(self, x):
        feat0 = self.layer0(x)
        feat1 = self.layer1(feat0)
        feat2 = self.layer2(feat1)
        # 返回三层特征：H/2, H/4, H/8
        return [feat0, feat1, feat2]

# =========================================================
# 模块 3: 注意力增强型分割头 (Attention Segmentation Head)
# 修改点：集成 CBAM 模块与 ASPP，通过多尺度特征融合实现“嗅觉”与“大视野”提升
# =========================================================
class SegmentationHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 引入注意力机制，自动学习哪些像素是干扰（标线），哪些是目标（异物）
        self.cbam = CBAMBlock(in_channels)
        
        # 【优化点】：用 ASPP 替换原有的 3x3 卷积，扩大感受野
        self.aspp = LightweightASPP(in_channels, 64)
        
        self.conv2 = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.cbam(x) # 特征筛选：压制白线，突出异物
        x = self.aspp(x) # 【优化点】：结合上下文宏观视野，抹平裂缝
        x = self.sigmoid(self.conv2(x))
        return x

# =========================================================
# 模块 4: 终极 E2E 师生网络 (Feature-Fusion Edition)
# =========================================================
class TeacherStudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Teacher (冻结)
        self.teacher = ResNetFeatureExtractor(pretrained=True)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # 2. Student (可训练)
        self.student = ResNetFeatureExtractor(pretrained=False)
        
        # 3. 分割头
        # 通道总数: Layer0(64) + Layer1(64) + Layer2(128) = 256
        self.segmenter = SegmentationHead(in_channels=256) 

    def forward(self, x):
        """
        [训练模式]
        """
        with torch.no_grad():
            t_feats = self.teacher(x)
        s_feats = self.student(x)
        
        # 【优化点】：计算三层残差，移除平方操作，使用 abs() 防止幅度爆炸，平方惩罚交由外面的 SmoothL1 处理
        diff0 = torch.abs(t_feats[0] - s_feats[0])  # H/2
        diff1 = torch.abs(t_feats[1] - s_feats[1])  # H/4
        diff2 = torch.abs(t_feats[2] - s_feats[2])  # H/8
        
        # 多尺度特征对齐：全部上采样到 H/2 分辨率
        diff1_up = F.interpolate(diff1, size=diff0.shape[2:], mode='bilinear', align_corners=False)
        diff2_up = F.interpolate(diff2, size=diff0.shape[2:], mode='bilinear', align_corners=False)
        
        # 拼接特征体：融合了从极微观到宏观的所有异常线索
        concat_features = torch.cat([diff0, diff1_up, diff2_up], dim=1) # [B, 256, H/2, W/2]
        
        # 通过注意力分割头
        pred_mask_mid_res = self.segmenter(concat_features)
        
        # 最终映射回原图尺寸
        pred_mask_full = F.interpolate(pred_mask_mid_res, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return t_feats, s_feats, pred_mask_full

    @torch.no_grad()
    def infer_mask(self, x, margin=0):
        """
        [推理模式]
        返回 0~1 原始概率图，交由后处理端 OpenCV 执行形态学双连击。
        """
        self.eval() 
        _, _, pred_mask = self.forward(x)
        
        if margin > 0:
            B, C, H, W = pred_mask.shape
            mask = torch.ones_like(pred_mask)
            mask[:, :, :margin, :] = 0
            mask[:, :, H-margin:, :] = 0
            mask[:, :, :, :margin] = 0
            mask[:, :, :, W-margin:] = 0
            pred_mask = pred_mask * mask
        
        return pred_mask