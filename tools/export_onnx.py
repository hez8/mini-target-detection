# export_onnx.py
import torch
import torch.nn as nn
from configs.default_config import Config
from models.distillation_net import TeacherStudentNet

class AnomalyInferenceWrapper(nn.Module):
    """
    封装模型推理，确保输出为单通道热力图。
    """
    def __init__(self, trained_student_path, backbone_name='resnet18'):
        super().__init__()
        self.cfg = Config()
        self.model = TeacherStudentNet(backbone_name=backbone_name)
        
        # 加载训练好的权重
        self.model.load_state_dict(torch.load(trained_student_path, map_location='cpu'))
        self.model.eval()

    def forward(self, x):
        # x shape: [B, 3, H, W]
        # 直接调用 infer_mask 获得 0~1 的概率图
        # 注意：ONNX 导出时，infer_mask 内部不应包含 margin 裁剪逻辑（或设为0）
        _, _, pred_mask = self.model(x)
        return pred_mask # [B, 1, H, W]

def export_to_onnx(weights_path, output_path="anomaly_detector.onnx", config=None):
    cfg = config if config is not None else Config()
    print(f"正在构建计算图 ({cfg.BACKBONE}) 并加载权重...")
    wrapper = AnomalyInferenceWrapper(weights_path, backbone_name=cfg.BACKBONE)
    wrapper.eval()
    
    # 构建一个符合输入尺寸的 Dummy Tensor
    dummy_input = torch.randn(1, 3, cfg.INPUT_SIZE[0], cfg.INPUT_SIZE[1])
    
    print(f"正在导出为 ONNX 模型: {output_path} ...")
    torch.onnx.export(
        wrapper, 
        dummy_input, 
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 
                      'output': {0: 'batch_size'}}
    )
    print("ONNX 导出成功！")

if __name__ == "__main__":
    export_to_onnx("checkpoints/best_student.pth")
