# models/distillation_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from configs.default_config import Config

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
        # 确保 channels // reduction 至少为 1
        mid_channels = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False)
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
        x = x * spatial_out
        
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
# 模块 2: 多骨干特征提取器工厂
# 支持: ResNet18, MobileNetV3, EfficientNet
# =========================================================
class FeatureExtractor(nn.Module):
    def __init__(self, backbone_name='resnet18', pretrained=False):
        super().__init__()
        self.backbone_name = backbone_name
        
        if backbone_name == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # H/2
            self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1) # H/4
            self.layer2 = resnet.layer2 # H/8
            self.out_channels = [64, 64, 128]
            
        elif backbone_name == 'mobilenet_v3_small':
            mnet = models.mobilenet_v3_small(pretrained=pretrained)
            features = list(mnet.features.children())
            self.layer0 = nn.Sequential(features[0]) # H/2, 16ch
            self.layer1 = nn.Sequential(*features[1:4]) # H/4, 24ch
            self.layer2 = nn.Sequential(*features[4:9]) # H/8, 48ch
            self.out_channels = [16, 24, 48]
            
        elif backbone_name == 'mobilenet_v3_large':
            mnet = models.mobilenet_v3_large(pretrained=pretrained)
            features = list(mnet.features.children())
            self.layer0 = nn.Sequential(features[0]) # H/2, 16ch
            self.layer1 = nn.Sequential(*features[1:4]) # H/4, 24ch
            self.layer2 = nn.Sequential(*features[4:7]) # H/8, 40ch
            self.out_channels = [16, 24, 40]
            
        elif backbone_name == 'efficientnet_b0':
            enet = models.efficientnet_b0(pretrained=pretrained)
            features = list(enet.features.children())
            self.layer0 = nn.Sequential(features[0], features[1]) # H/2, 16ch
            self.layer1 = nn.Sequential(features[2]) # H/4, 24ch
            self.layer2 = nn.Sequential(features[3]) # H/8, 40ch
            self.out_channels = [16, 24, 40]
            
        elif backbone_name in ['yolo11n', 'yolo11s']:
            try:
                from ultralytics import YOLO
            except ImportError:
                raise ImportError("请运行 'pip install ultralytics' 以支持 YOLO 系列骨干网络")
            
            # 加载模型结构，如果 pretrained=True 则自动下载权重
            yolo_model = YOLO(f"{backbone_name}.pt")
            # 提取其内部的 nn.Sequential backbone
            yolo_backbone = yolo_model.model.model
            
            # 根据 YOLO11 架构，提取各尺度特征层
            # P1 (H/2): Layer 0 (Conv)
            self.layer0 = yolo_backbone[0]
            # P2 (H/4): Layer 1 (Conv) + Layer 2 (C3k2)
            self.layer1 = nn.Sequential(yolo_backbone[1], yolo_backbone[2])
            # P3 (H/8): Layer 3 (Conv) + Layer 4 (C3k2)
            self.layer2 = nn.Sequential(yolo_backbone[3], yolo_backbone[4])
            
            if backbone_name == 'yolo11n':
                self.out_channels = [16, 64, 128]
            else: # yolo11s
                self.out_channels = [32, 128, 256]
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x):
        feat0 = self.layer0(x)
        feat1 = self.layer1(feat0)
        feat2 = self.layer2(feat1)
        return [feat0, feat1, feat2]

# =========================================================
# 模块 3: 注意力增强型分割头 (Attention Segmentation Head)
# =========================================================
class SegmentationHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cbam = CBAMBlock(in_channels)
        self.aspp = LightweightASPP(in_channels, 64)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.cbam(x) 
        x = self.aspp(x) 
        x = self.sigmoid(self.conv2(x))
        return x

# =========================================================
# 模块 4: 终极 E2E 师生网络 (Configurable Edition)
# =========================================================
class TeacherStudentNet(nn.Module):
    def __init__(self, backbone_name=None):
        super().__init__()
        # 如果未指定，则从 Config 中自动读取
        if backbone_name is None:
            backbone_name = getattr(Config, 'BACKBONE', 'resnet18')
            
        # 1. Teacher (冻结)
        self.teacher = FeatureExtractor(backbone_name=backbone_name, pretrained=True)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # 2. Student (可训练)
        self.student = FeatureExtractor(backbone_name=backbone_name, pretrained=False)
        
        # 3. 分割头
        in_channels = sum(self.student.out_channels)
        self.segmenter = SegmentationHead(in_channels=in_channels) 

    def forward(self, x):
        with torch.no_grad():
            t_feats = self.teacher(x)
        s_feats = self.student(x)
        
        diff0 = torch.abs(t_feats[0] - s_feats[0])  # H/2
        diff1 = torch.abs(t_feats[1] - s_feats[1])  # H/4
        diff2 = torch.abs(t_feats[2] - s_feats[2])  # H/8
        
        # 多尺度特征对齐：全部上采样到 H/2 分辨率
        diff1_up = F.interpolate(diff1, size=diff0.shape[2:], mode='bilinear', align_corners=False)
        diff2_up = F.interpolate(diff2, size=diff0.shape[2:], mode='bilinear', align_corners=False)
        
        # 拼接特征体
        concat_features = torch.cat([diff0, diff1_up, diff2_up], dim=1) 
        
        # 通过注意力分割头
        pred_mask_mid_res = self.segmenter(concat_features)
        
        # 最终映射回原图尺寸
        pred_mask_full = F.interpolate(pred_mask_mid_res, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return t_feats, s_feats, pred_mask_full

    @torch.no_grad()
    def infer_mask(self, x, margin=0):
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
