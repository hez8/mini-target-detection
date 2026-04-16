# inference_map.py
import cv2
import torch
import numpy as np
import time
import queue
import threading
import os
import glob

from configs.default_config import Config
from models.distillation_net import TeacherStudentNet
from torchvision import transforms

# 强制性能优化：解除 OpenCV 与 PyTorch 的线程锁
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def video_reader_thread(video_source, q_frames, stop_event):
    """【生产者线程】负责高速读取视频帧"""
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[-] 无法打开视频源: {video_source}")
        stop_event.set()
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            stop_event.set()
            break
        # 限制队列长度，防止内存溢出
        if q_frames.qsize() < 3:
            q_frames.put(frame)
        else:
            time.sleep(0.01)
    cap.release()

def process_single_map_video(video_source, model, transform, cfg, weights_path, save_video):
    """【消费者逻辑】处理单个视频并生成原始热力图"""
    q_frames = queue.Queue(maxsize=5)
    stop_event = threading.Event()
    
    reader_t = threading.Thread(target=video_reader_thread, args=(video_source, q_frames, stop_event))
    reader_t.daemon = True
    reader_t.start()

    # ---------------- 录像输出路径处理 (与 inference 一致) ----------------
    video_writer = None
    output_path = None
    if save_video:
        output_dir = os.path.join("data", "test_videos_result")
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取模型文件夹的后六位
        model_dir_name = os.path.basename(os.path.dirname(weights_path))
        model_suffix = model_dir_name[-6:] if len(model_dir_name) >= 6 else model_dir_name
        
        # 组合文件名，前缀改为 map_
        if isinstance(video_source, str) and not video_source.isdigit():
            src_name = os.path.splitext(os.path.basename(video_source))[0]
            output_filename = f"map_{src_name}_{model_suffix}.mp4"
        else:
            output_filename = f"map_camera_{video_source}_{model_suffix}.mp4"
            
        output_path = os.path.join(output_dir, output_filename)
    # ---------------------------------------------------------------------

    print(f"[*] 正在生成热力图: {video_source}")

    while not stop_event.is_set() or not q_frames.empty():
        try:
            frame = q_frames.get(timeout=0.5)
        except queue.Empty:
            continue
            
        if save_video and video_writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))

        # ---------- A. 网络推理 ----------
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb_frame).unsqueeze(0).to(cfg.DEVICE)

        with torch.no_grad():
            # 调用新架构模型的接口，获取 0~1 的原始浮点概率图
            raw_prob_tensor = model.infer_mask(input_tensor, margin=getattr(cfg, 'EDGE_IGNORE_MARGIN', 0))
            
        prob_map = raw_prob_tensor.squeeze().cpu().numpy()

        # ---------- B. 热力图可视化 ----------
        # 1. 线性映射到 0-255
        heatmap_gray = (prob_map * 255).astype(np.uint8)
        
        # 2. 应用伪彩色映射 (JET 模式：冷色蓝到暖色红)
        heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
        
        # 3. 将热力图与原图按 6:4 比例融合，方便观察异物在路面的精确位置
        merged_display = cv2.addWeighted(frame, 0.4, heatmap_color, 0.6, 0)

        # 4. 辅助信息渲染
        max_prob = np.max(prob_map)
        cv2.putText(merged_display, f"MAX CONF: {max_prob:.4f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(merged_display, "DIAGNOSTIC MODE: RAW HEATMAP", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        if video_writer is not None:
            video_writer.write(merged_display)
            
        q_frames.task_done()

    if video_writer is not None:
        video_writer.release()
    print(f"[+] 导出完成: {output_path}")

def main_consumer_pipeline(video_source='0', weights_path='checkpoints/best_student.pth', save_video=True, config=None):
    """批量热力图生成调度中心"""
    cfg = config if config is not None else Config()
    
    # 1. 加载模型 (仅加载一次)
    print(f"[*] 正在加载真·端到端模型 ({cfg.BACKBONE})，设备: {cfg.DEVICE}")
    model = TeacherStudentNet(backbone_name=cfg.BACKBONE).to(cfg.DEVICE)
    
    if not os.path.exists(weights_path):
        print(f"[-] 错误：找不到权重文件 {weights_path}")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=cfg.DEVICE))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. 解析任务列表 (支持文件夹遍历)
    video_tasks = []
    if os.path.isdir(video_source):
        supported_exts = ('.mp4', '.avi', '.mov', '.mkv')
        for f in os.listdir(video_source):
            if f.lower().endswith(supported_exts):
                video_tasks.append(os.path.join(video_source, f))
        print(f"[*] 发现文件夹任务，共计 {len(video_tasks)} 个视频")
    else:
        video_tasks.append(video_source)

    # 3. 顺序处理任务
    for vs in video_tasks:
        try:
            process_single_map_video(vs, model, transform, cfg, weights_path, save_video)
        except KeyboardInterrupt:
            break

    print("[*] 所有热力图生成任务已结束。")

if __name__ == "__main__":
    # 如果需要独立运行，可以在此处添加解析逻辑，通常建议通过 main.py 调用
    pass