# models/distillation_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class FeatureExtractor(nn.Module):
    """提取 ResNet 前三层的特征图，用于多尺度比对"""
    def __init__(self, pretrained=False):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = resnet18(weights=weights)
        
        self.layer1 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool, base_model.layer1)
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        
    def forward(self, x):
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        return [feat1, feat2, feat3]

class TeacherStudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Teacher: 预训练且冻结权重
        self.teacher = FeatureExtractor(pretrained=True)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval() # 永远保持 eval 模式
            
        # Student: 随机初始化，需训练
        self.student = FeatureExtractor(pretrained=False)
        
    def forward(self, x):
        with torch.no_grad():
            t_feats = self.teacher(x)
        s_feats = self.student(x)
        return t_feats, s_feats

    def compute_anomaly_map(self, t_feats, s_feats, input_size):
        """计算师生网络特征差异，并融合为一张异常图"""
        anomaly_map = 0
        for t_feat, s_feat in zip(t_feats, s_feats):
            # 计算 L2 距离 (C 通道求均值)
            diff = torch.mean((t_feat - s_feat) ** 2, dim=1, keepdim=True)
            # 上采样到输入图像大小
            diff_up = F.interpolate(diff, size=input_size, mode='bilinear', align_corners=False)
            anomaly_map += diff_up
            
        return anomaly_map # Shape: [B, 1, H, W]