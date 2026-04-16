# inference.py
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
from modules.spatio_filter import SpatioTemporalFilter
from torchvision import transforms

def video_reader_thread(video_source, q_frames, stop_event):
    """【生产者线程】负责从视频流中高速读取画面"""
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[-] 无法打开视频源: {video_source}")
        stop_event.set()
        return

    print(f"[+] 成功打开视频源: {video_source}")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[*] 视频流读取完毕。")
            stop_event.set()
            break
            
        # 保持队列中最多只有少量最新帧，防止内存爆炸和延迟累积
        if q_frames.qsize() < 3:
            q_frames.put(frame)
        else:
            time.sleep(0.01)
            
    cap.release()

def process_single_video(video_source, model, spatio_filter, transform, cfg, weights_path, save_video):
    """处理单个视频流的核心消费者逻辑"""
    q_frames = queue.Queue(maxsize=5)
    stop_event = threading.Event()
    
    # 启动读取线程
    reader_t = threading.Thread(target=video_reader_thread, args=(video_source, q_frames, stop_event))
    reader_t.daemon = True
    reader_t.start()

    # ---------------- 录像输出路径处理 ----------------
    video_writer = None
    output_path = None
    if save_video:
        output_dir = os.path.join("data", "test_videos_result")
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取模型文件夹的后六位用于版本追踪
        model_dir_name = os.path.basename(os.path.dirname(weights_path))
        model_suffix = model_dir_name[-6:] if len(model_dir_name) >= 6 else model_dir_name
        
        if isinstance(video_source, str) and not video_source.isdigit():
            src_name = os.path.splitext(os.path.basename(video_source))[0]
            output_filename = f"output_{src_name}_{model_suffix}.mp4"
        else:
            output_filename = f"output_camera_{video_source}_{model_suffix}.mp4"
            
        output_path = os.path.join(output_dir, output_filename)
    # ------------------------------------------------------------

    frame_count = 0
    fps_start_time = time.time()
    
    # EMA 时序平滑用的历史概率图变量
    prev_prob_map = None 
    
    # 只要没收到停止信号，或者队列里还有剩余帧，就继续处理
    while not stop_event.is_set() or not q_frames.empty():
        try:
            frame = q_frames.get(timeout=0.5)
        except queue.Empty:
            continue
            
        if save_video and video_writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))

        display_frame = frame.copy()

        # ---------- A. 数据预处理 ----------
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb_frame).unsqueeze(0).to(cfg.DEVICE)

        # ---------- B. E2E GPU 掩膜推理 ----------
        with torch.no_grad():
            margin_val = getattr(cfg, 'EDGE_IGNORE_MARGIN', 0)
            # 【核心对接】：接收 distillation_net 修改后吐出的 0~1 浮点概率图
            raw_prob_tensor = model.infer_mask(input_tensor, margin=margin_val)
            
        current_prob_map = raw_prob_tensor.squeeze().cpu().numpy()

        # ---------- C. 时序平滑 (EMA) ----------
        if prev_prob_map is None:
            prev_prob_map = current_prob_map
        else:
            # 融合上一帧残影，极大地消除边缘闪烁
            current_prob_map = 0.6 * current_prob_map + 0.4 * prev_prob_map
        prev_prob_map = current_prob_map

        # ---------- D. 严格阈值切割与形态学双连击 ----------
        # 1. 以 0.65 的高阈值进行切割
        binary_mask = np.where(current_prob_map > 0.65, 255, 0).astype(np.uint8)
        
        # 2. 开运算 (Open)：抹杀背景中的孤立高频噪点
        kernel_open = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
        
        # 3. 闭运算 (Close)：填补真实大异物内部的特征断裂和空洞
        kernel_close = np.ones((21, 21), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)

        # ---------- E. 提取连通域 ----------
        targets = spatio_filter.extract_targets(binary_mask)

        # ---------- F. 高级渲染 (GrabCut 优化) ----------
        overlay = display_frame.copy() 
        alpha = 0.5                    
        mask_color = (0, 0, 255)       

        for t in targets:
            x, y, w, h = t['bbox']
            
            # 【自适应 Padding】：物体越大，向外扩展寻找背景的范围也应该越大
            pad_x = max(15, int(w * 0.2))
            pad_y = max(15, int(h * 0.2))
            
            y1, y2 = max(0, y - pad_y), min(display_frame.shape[0], y + h + pad_y)
            x1, x2 = max(0, x - pad_x), min(display_frame.shape[1], x + w + pad_x)
            
            roi_color = display_frame[y1:y2, x1:x2]
            roi_dl_mask = binary_mask[y1:y2, x1:x2]
            
            # 初始化 GrabCut 蒙版
            gc_mask = np.zeros(roi_color.shape[:2], np.uint8)
            gc_mask[:] = cv2.GC_PR_BGD 
            gc_mask[roi_dl_mask > 0] = cv2.GC_PR_FGD 
            
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            
            # 运行 GrabCut 精细化边缘
            try:
                cv2.grabCut(roi_color, gc_mask, None, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_MASK)
            except cv2.error:
                pass # ROI 太小可能失败，失败则回退到原生 Mask
                
            refined_anomaly_pixels = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), True, False)
            
            # 对精确的像素点上色
            overlay_roi = overlay[y1:y2, x1:x2]
            overlay_roi[refined_anomaly_pixels] = mask_color
            overlay[y1:y2, x1:x2] = overlay_roi
            
            # 绘制 ID 和告警日志
            cv2.putText(display_frame, f"ID:{t['id']}", (x, max(15, y-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
        # 半透明混合
        cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)

        # ---------- G. 帧率统计与输出 ----------
        frame_count += 1
        elapsed = time.time() - fps_start_time
        fps = frame_count / elapsed
        cv2.putText(display_frame, f"E2E FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 后台写入视频流
        if video_writer is not None:
            video_writer.write(display_frame)
            
        q_frames.task_done()

    if video_writer is not None:
        video_writer.release()
    print(f"[+] 视频处理完毕: {video_source} -> {output_path}")

def main_consumer_pipeline(video_source='0', weights_path='checkpoints/best_student.pth', save_video=False):
    """【调度中心】批量/单文件推理，全局只加载一次模型"""
    cfg = Config()
    
    # 1. 统一加载模型，避免批量测试时重复加载
    print(f"[*] 正在加载真·端到端模型，设备: {cfg.DEVICE}")
    model = TeacherStudentNet().to(cfg.DEVICE)
    
    if not os.path.exists(weights_path):
        print(f"[致命错误] 找不到权重文件: {weights_path}")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=cfg.DEVICE))
    model.eval() 
    print("[+] 模型加载与初始化完毕。")

    spatio_filter = SpatioTemporalFilter(cfg)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. 判断输入类型：文件夹 or 单文件/摄像头
    video_tasks = []
    if os.path.isdir(video_source):
        # 搜集文件夹下所有的视频文件
        supported_exts = ('.mp4', '.avi', '.mov', '.mkv')
        for f in os.listdir(video_source):
            if f.lower().endswith(supported_exts):
                video_tasks.append(os.path.join(video_source, f))
        
        if not video_tasks:
            print(f"[-] 在文件夹 {video_source} 中没有找到支持的视频文件。")
            return
        print(f"[*] 检测到文件夹输入，共找到 {len(video_tasks)} 个视频等待批量处理...")
    else:
        # 单文件或摄像头
        video_tasks.append(video_source)

    # 3. 逐个执行推理任务
    for idx, vs in enumerate(video_tasks):
        if len(video_tasks) > 1:
            print(f"\n[>>>] 正在处理第 {idx+1}/{len(video_tasks)} 个视频: {vs}")
        else:
            print(f"[*] 开始实时推理 (终端按 Ctrl+C 强制退出)...")
            
        try:
            process_single_video(
                video_source=vs, 
                model=model, 
                spatio_filter=spatio_filter, 
                transform=transform, 
                cfg=cfg, 
                weights_path=weights_path, 
                save_video=save_video
            )
        except KeyboardInterrupt:
            print("\n[*] 检测到用户中断信号，已提前终止所有后续任务。")
            break

    print("\n[*] === 所有推理任务执行完毕！ ===")