# inference_async.py
import cv2
import torch
import numpy as np
import threading
import queue
import time
from torchvision import transforms

# 导入配置与自定义的核心模块
from configs.default_config import Config
from models.distillation_net import TeacherStudentNet
from modules.ego_motion import EgoMotionCompensator
from modules.spatio_filter import SpatioTemporalFilter
from modules.patch_processor import ImagePatcher

# ==========================================
# 全局队列与事件控制 (多线程共享)
# ==========================================
# maxsize=3 防止生产者过快导致内存/显存爆满
q_aligned_frames = queue.Queue(maxsize=3)
q_anomaly_maps = queue.Queue(maxsize=3)
stop_event = threading.Event()

# ==========================================
# 线程 A：视频读取与全局运动补偿 (纯 CPU)
# ==========================================
def producer_thread(video_source, cfg, compensator):
    """负责视频流解码、维护滑动窗口和全局图像对齐"""
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频源: {video_source}")
        stop_event.set()
        return

    frame_buffer = []
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[提示] 视频流结束或读取失败。")
            stop_event.set()
            break
            
        frame_buffer.append(frame)
        if len(frame_buffer) > cfg.WINDOW_SIZE:
            frame_buffer.pop(0)
            
        if len(frame_buffer) == cfg.WINDOW_SIZE:
            # 耗时的特征匹配与仿射变换，在独立的 CPU 线程中完成
            aligned_frames = compensator.align_buffer(frame_buffer)
            center_frame = frame_buffer[compensator.center_idx].copy()
            
            # 阻塞放入队列，如果下游 GPU 处理不过来，这里会自动停下等待
            q_aligned_frames.put((center_frame, aligned_frames))

    cap.release()

# ==========================================
# 线程 B：切图、拼接与深度学习推理 (榨干 GPU)
# ==========================================
def gpu_worker_thread(cfg, model, patcher):
    """负责大图切块、打包送入 GPU 提取异常特征、并无缝拼接回原分辨率"""
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    MINI_BATCH_SIZE = 16 # 控制显存占用量
    
    while not stop_event.is_set():
        try:
            # 设置超时，防止主程序发出退出指令时死锁
            center_frame, aligned_frames = q_aligned_frames.get(timeout=1)
        except queue.Empty:
            continue
            
        original_shape = center_frame.shape
        full_anomaly_maps = []
        
        with torch.no_grad():
            for a_frame in aligned_frames:
                # 1. 大图切分成具有重叠率的 Patch 集合
                patches, coords = patcher.crop_to_patches(a_frame)
                patch_anomaly_maps = []
                
                # 2. 分批次进行推理解除显存压力
                for i in range(0, len(patches), MINI_BATCH_SIZE):
                    batch_patches = patches[i:i + MINI_BATCH_SIZE]
                    
                    # 预处理：BGR转RGB 并归一化
                    batch_tensors = [preprocess(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)) for p in batch_patches]
                    input_batch = torch.stack(batch_tensors).to(cfg.DEVICE, non_blocking=True)
                    
                    # 师生网络提取特征差异
                    t_feats, s_feats = model(input_batch)
                    a_map_batch = model.compute_anomaly_map(t_feats, s_feats, input_size=cfg.INPUT_SIZE)
                    
                    # 转移回 CPU numpy 以释放显存
                    a_map_batch_np = a_map_batch.squeeze(1).cpu().numpy()
                    if len(a_map_batch_np.shape) == 2:
                        a_map_batch_np = np.expand_dims(a_map_batch_np, axis=0)
                        
                    patch_anomaly_maps.extend([a_map_batch_np[j] for j in range(a_map_batch_np.shape[0])])
                
                # 3. 拼接回全分辨率异常图 (消除边缘接缝伪影)
                stitched_map = patcher.stitch_anomaly_maps(patch_anomaly_maps, coords, original_shape)
                full_anomaly_maps.append(stitched_map)
                
        # 将推理结果送入下游队列供主线程画图
        q_anomaly_maps.put((center_frame, full_anomaly_maps))
        q_aligned_frames.task_done()

# ==========================================
# 主线程：追踪过滤、可视化展示与录像持久化
# ==========================================
def main_consumer_pipeline(video_source=0, weights_path='checkpoints/best_student.pth', save_video=False):
    cfg = Config()
    
    print("[1/3] 正在初始化硬件与算法模块...")
    
    # 1. 初始化深度学习模型
    model = TeacherStudentNet().to(cfg.DEVICE)
    try:
        model.student.load_state_dict(torch.load(weights_path, map_location=cfg.DEVICE))
        print(f"成功加载模型权重: {weights_path}")
    except Exception as e:
        print(f"[警告] 无法加载权重文件 {weights_path}，将使用随机初始化的网络进行系统测试。({e})")
    model.eval()
    
    # 2. 初始化核心外围模块
    compensator = EgoMotionCompensator(window_size=cfg.WINDOW_SIZE)
    spatio_filter = SpatioTemporalFilter(cfg)
    
    # 切图器：保证有 50% 的重叠率以消除边缘效应
    stride_y = cfg.INPUT_SIZE[0] // 2
    stride_x = cfg.INPUT_SIZE[1] // 2
    patcher = ImagePatcher(patch_size=cfg.INPUT_SIZE, stride=(stride_y, stride_x))
    
    # 3. 初始化视频写入器 (如果开启保存)
    video_writer = None
    if save_video:
        cap_temp = cv2.VideoCapture(video_source)
        if cap_temp.isOpened():
            width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap_temp.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps != fps: fps = 25.0
            cap_temp.release()
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = "output_detection.mp4"
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"检测结果将被实时录制并保存至: {output_path}")

    print("[2/3] 启动多线程异步流水线...")
    t_prod = threading.Thread(target=producer_thread, args=(video_source, cfg, compensator))
    t_gpu = threading.Thread(target=gpu_worker_thread, args=(cfg, model, patcher))
    
    # 设为守护线程，主线程崩溃时一同结束
    t_prod.daemon = True
    t_gpu.daemon = True
    t_prod.start()
    t_gpu.start()
    
    fps_start_time = time.time()
    frame_count = 0

    print("[3/3] 进入实时推理主循环 (点击画面按 'q' 键退出)...")
    
    while not stop_event.is_set():
        try:
            display_frame, full_anomaly_maps = q_anomaly_maps.get(timeout=1)
        except queue.Empty:
            continue
            
        # 1. 时空滤波：结合 CentroidTracker 剔除高频树叶抖动，锁定真实的微小异物
        targets = spatio_filter.extract_targets(full_anomaly_maps)
        
        # 2. 绘制结果框与写入告警日志
        for t in targets:
            x, y, w, h = t['bbox']
            
            # 使用目标的 Tracker ID 保持绘制标识的一致性
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(display_frame, f"ID:{t['id']} Area:{t['area']}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # 本地持久化留档
            with open("alert_log.txt", "a") as f:
                f.write(f"Alert: FrameID:{frame_count}, TargetID:{t['id']}, Center:({x+w//2},{y+h//2}), Area:{t['area']}\n")
        
        # 3. 计算并绘制系统端到端 FPS
        frame_count += 1
        elapsed = time.time() - fps_start_time
        fps = frame_count / elapsed
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 4. 显示画面与写入录像文件
        cv2.imshow("Dynamic Small Target Anomaly Detection", display_frame)
        
        if video_writer is not None:
            video_writer.write(display_frame)
            
        q_anomaly_maps.task_done()
        
        # 5. 监听退出按键
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n收到退出指令,正在通知底层工作线程停止...")
            stop_event.set()

    print("正在等待工作线程安全退出 (可能需要几秒钟)...")
    t_prod.join(timeout=2)
    t_gpu.join(timeout=2)
    
    if video_writer is not None:
        video_writer.release()
        
    cv2.destroyAllWindows()
    print("推理引擎已彻底关闭。")

if __name__ == "__main__":
    # 如果单独运行此文件，默认使用本地 0 号摄像头进行测试，不开启硬盘录像
    main_consumer_pipeline(video_source=0, weights_path='checkpoints/best_student.pth', save_video=False)