# main.py
import argparse
import os
import cv2

# ====== 确保这里的导入名与你的文件名完全一致 ======
from train import train_pipeline
from inference import main_consumer_pipeline  # 确保是从 inference 导入
from export_onnx import export_to_onnx
from configs.default_config import Config
from inference_map import main_consumer_pipeline as map_pipeline  # 导入热力图诊断模块

def get_latest_model_path(base_dir="checkpoints", target_model="best_student.pth"):
    """自动寻找最新的模型权重"""
    if not os.path.exists(base_dir):
        return None
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        return None
    # 按照修改时间排序，获取最新的文件夹
    latest_dir = max(subdirs, key=os.path.getmtime)
    latest_model_path = os.path.join(latest_dir, target_model)
    return latest_model_path if os.path.exists(latest_model_path) else None

def parse_args():
    parser = argparse.ArgumentParser(description="动态背景微小目标检测系统 (Dynamic Small Target Anomaly Detection)")
    
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer', 'export', 'map'],
                        help="运行模式：train(无监督训练), infer(实时推理), export(导出ONNX)")
    
    parser.add_argument('--video_source', type=str, default='0',
                        help="推理模式下的视频源，可以是 0 (摄像头) 或视频文件路径")
    
    parser.add_argument('--save_video', action='store_true',
                        help="推理模式下，是否将结果保存为 mp4 视频")
                        
    # [核心修改]：默认值从固定路径改为 'auto'
    parser.add_argument('--weights', type=str, default='auto',
                        help="推理或导出时使用的模型权重路径。默认 'auto' 会自动寻找 checkpoints 下最新训练的模型。")

    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config()

    if args.mode == 'train':
        print(">>> 启动无监督训练流水线...")
        train_pipeline()

    elif args.mode == 'infer':
        print(">>> 启动多线程异步推理流水线...")
        
        # 1. 自动权重寻路逻辑
        weights_path = args.weights
        if weights_path == 'auto':
            print("[*] 正在自动搜索最新训练的模型...")
            weights_path = get_latest_model_path()
            if weights_path is None:
                print("[-] 致命错误：没有找到模型！请先运行训练。")
                return
            print(f"[+] 自动锁定最新模型: {weights_path}")

        # 2. 处理设备号 (0) 和字符串路径的区别
        source = int(args.video_source) if args.video_source.isdigit() else args.video_source
        
        # 3. 将视频源和权重路径传递给 inference_async
        main_consumer_pipeline(video_source=source, weights_path=weights_path, save_video=args.save_video)

    elif args.mode == 'map':
        # [新增] 热力图诊断逻辑
        print(">>> 启动热力图可视化诊断模式...")
        print("[提示] 红色代表网络判定的高风险异常区域，蓝色为安全区域。")
        # 1. 自动权重寻路逻辑
        weights_path = args.weights
        if weights_path == 'auto':
            print("[*] 正在自动搜索最新训练的模型...")
            weights_path = get_latest_model_path()
            if weights_path is None:
                print("[-] 致命错误：没有找到模型！请先运行训练。")
                return
            print(f"[+] 自动锁定最新模型: {weights_path}")

        # 2. 处理设备号 (0) 和字符串路径的区别
        source = int(args.video_source) if args.video_source.isdigit() else args.video_source
        
        map_pipeline(video_source=source, weights_path=weights_path, save_video=args.save_video)

    elif args.mode == 'export':
        print(f">>> 正在将 PyTorch 模型导出为 ONNX...")
        
        # 导出模式同样支持自动寻路
        weights_path = args.weights
        if weights_path == 'auto':
            weights_path = get_latest_model_path()
            if weights_path is None:
                print("[-] 致命错误：没有找到模型！请先运行训练。")
                return

        print(f"[*] 权重来源: {weights_path}")
        if not os.path.exists(weights_path):
            print("错误：未找到权重文件！请先运行 --mode train")
            return
        export_to_onnx(weights_path, output_path="anomaly_detector.onnx")

if __name__ == "__main__":
    main()