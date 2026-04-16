# export_onnx.py
import torch
import torch.nn as nn
from configs.default_config import Config
from models.distillation_net import TeacherStudentNet

class AnomalyInferenceWrapper(nn.Module):
    """
    为了部署优化，我们将 Teacher、Student 和 Loss 计算封装在一个图(Graph)内。
    这样可以避免把高维特征图传回 CPU 再做计算。
    """
    def __init__(self, trained_student_path):
        super().__init__()
        self.cfg = Config()
        self.model = TeacherStudentNet()
        
        # 加载训练好的 Student 权重
        self.model.student.load_state_dict(torch.load(trained_student_path, map_location='cpu'))
        self.model.eval()

    def forward(self, x):
        # x shape: [B, 3, H, W]
        t_feats, s_feats = self.model(x)
        
        # 直接在计算图内完成特征差异计算
        anomaly_map = 0
        for t_feat, s_feat in zip(t_feats, s_feats):
            # 计算 L2 距离
            diff = torch.mean((t_feat - s_feat) ** 2, dim=1, keepdim=True)
            # 使用双线性插值上采样到输入 Patch 大小
            diff_up = nn.functional.interpolate(diff, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            anomaly_map += diff_up
            
        return anomaly_map # 直接输出热力图 [B, 1, H, W]

def export_to_onnx(weights_path, output_path="anomaly_detector.onnx"):
    print("正在构建计算图并加载权重...")
    wrapper = AnomalyInferenceWrapper(weights_path)
    wrapper.eval()
    
    # 构建一个符合输入尺寸的 Dummy Tensor (模拟 Batch Size = 16 的 Patch)
    cfg = Config()
    dummy_input = torch.randn(16, 3, cfg.INPUT_SIZE[0], cfg.INPUT_SIZE[1])
    
    print(f"正在导出为 ONNX 模型: {output_path} ...")
    torch.onnx.export(
        wrapper, 
        dummy_input, 
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input_patches'],
        output_names=['anomaly_map'],
        # 允许动态 Batch Size，因为边缘画面切出来的 Patch 数量可能变化
        dynamic_axes={'input_patches': {0: 'batch_size'}, 
                      'anomaly_map': {0: 'batch_size'}}
    )
    print("ONNX 导出成功！后续可导入 TensorRT 进行极致加速。")

if __name__ == "__main__":
    # 请替换为实际的最好模型权重路径
    export_to_onnx("checkpoints/best_student.pth")