# inference.py
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
# 线程 A：视频读取与全局运动补偿 (CPU)
# ==========================================
def producer_thread(video_source, cfg, compensator):
    """负责视频解码、维护窗口和全局对齐"""
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
            # 执行耗时的特征匹配与仿射变换
            aligned_frames = compensator.align_buffer(frame_buffer)
            center_frame = frame_buffer[compensator.center_idx].copy()
            
            # 阻塞放入队列，如果 GPU 处理不过来，这里会停下等待
            q_aligned_frames.put((center_frame, aligned_frames))

    cap.release()

# ==========================================
# 线程 B：切图、拼接与深度学习推理 (GPU)
# ==========================================
def gpu_worker_thread(cfg, model, patcher):
    """负责切图、打包送入 GPU 推理、并无缝拼接回原图"""
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    MINI_BATCH_SIZE = 16 # 防止 1080P 切出太多块导致显存 OOM
    
    while not stop_event.is_set():
        try:
            # 设置超时，防止主程序退出时死锁
            center_frame, aligned_frames = q_aligned_frames.get(timeout=1)
        except queue.Empty:
            continue
            
        original_shape = center_frame.shape
        full_anomaly_maps = []
        
        with torch.no_grad():
            for a_frame in aligned_frames:
                # 1. 大图切小块
                patches, coords = patcher.crop_to_patches(a_frame)
                patch_anomaly_maps = []
                
                # 2. 分批次推理
                for i in range(0, len(patches), MINI_BATCH_SIZE):
                    batch_patches = patches[i:i + MINI_BATCH_SIZE]
                    
                    # 预处理：BGR 转 RGB 并归一化
                    batch_tensors = [preprocess(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)) for p in batch_patches]
                    input_batch = torch.stack(batch_tensors).to(cfg.DEVICE, non_blocking=True)
                    
                    # 网络提取异常特征图
                    t_feats, s_feats = model(input_batch)
                    a_map_batch = model.compute_anomaly_map(t_feats, s_feats, input_size=cfg.INPUT_SIZE, k_sigma=cfg.K_SIGMA, margin=cfg.SMOOTH_KERNAEL)
                    
                    # 转移回 CPU numpy 释放显存
                    a_map_batch_np = a_map_batch.squeeze(1).cpu().numpy()
                    if len(a_map_batch_np.shape) == 2:
                        a_map_batch_np = np.expand_dims(a_map_batch_np, axis=0)
                        
                    patch_anomaly_maps.extend([a_map_batch_np[j] for j in range(a_map_batch_np.shape[0])])
                
                # 3. 拼接回全分辨率异常图
                stitched_map = patcher.stitch_anomaly_maps(patch_anomaly_maps, coords, original_shape)
                full_anomaly_maps.append(stitched_map)
                
        # 将推理结果送入下游队列
        q_anomaly_maps.put((center_frame, full_anomaly_maps))
        q_aligned_frames.task_done()

# ==========================================
# 主线程：追踪过滤、可视化与录像持久化
# ==========================================
def main_consumer_pipeline(video_source=0, weights_path='checkpoints/best_student.pth', save_video=False):
    cfg = Config()
    
    print("[1/3] 正在初始化硬件与算法模块...")
    
    # 初始化模型
    model = TeacherStudentNet().to(cfg.DEVICE)
    try:
        model.student.load_state_dict(torch.load(weights_path, map_location=cfg.DEVICE))
        print(f"成功加载权重: {weights_path}")
    except Exception as e:
        print(f"[警告] 无法加载权重文件 {weights_path}，将使用随机初始化权重进行测试。({e})")
    model.eval()
    
    # 初始化外围模块
    compensator = EgoMotionCompensator(window_size=cfg.WINDOW_SIZE)
    spatio_filter = SpatioTemporalFilter(cfg)
    
    # 保证切片有 50% 的重叠率以消除边缘伪影
    stride_y = cfg.INPUT_SIZE[0] // 2
    stride_x = cfg.INPUT_SIZE[1] // 2
    patcher = ImagePatcher(patch_size=cfg.INPUT_SIZE, stride=(stride_y, stride_x))
    
    # 初始化视频写入器 (如果需要保存)
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
    
    t_prod.start()
    t_gpu.start()
    
    fps_start_time = time.time()
    frame_count = 0

    print("[3/3] 进入实时推理主循环 (按 'q' 退出)...")
    
    while not stop_event.is_set():
        try:
            display_frame, full_anomaly_maps = q_anomaly_maps.get(timeout=1)
        except queue.Empty:
            continue

        # 提取当前帧的原始高精度二值化 Mask，并确保格式正确
        current_mask = full_anomaly_maps[-1] if isinstance(full_anomaly_maps, list) else full_anomaly_maps
        if hasattr(current_mask, 'cpu'):
            current_mask = current_mask.cpu().numpy()
        current_mask = current_mask.astype(np.uint8)

        # 1. 结合 CentroidTracker 进行极值剔除，锁定真实目标
        targets = spatio_filter.extract_targets(full_anomaly_maps)
        
        # 2. 绘制结果与记录日志， 采用半透明 Mask 叠加绘制法
        # =======================================================
        # 采用半透明 Mask 叠加绘制法 + GrabCut 边缘精准吸附
        # =======================================================
        overlay = display_frame.copy() 
        alpha = 0.5                    # 透明度
        mask_color = (0, 0, 255)       # 纯红色

        for t in targets:
            x, y, w, h = t['bbox']
            
            # 1. 给 Bounding Box 增加一点 Padding (外扩)，非常关键！
            # GrabCut 需要看到目标周围的一点真实地板，才能学习什么是“背景”
            pad = 15
            y1, y2 = max(0, y - pad), min(display_frame.shape[0], y + h + pad)
            x1, x2 = max(0, x - pad), min(display_frame.shape[1], x + w + pad)
            
            # 2. 提取原图的 RGB ROI 和 深度学习输出的粗略 Mask ROI
            roi_color = display_frame[y1:y2, x1:x2]
            roi_dl_mask = current_mask[y1:y2, x1:x2]
            
            # 3. 初始化 GrabCut 需要的蒙版
            # cv2.GC_BGD(0): 确定背景, cv2.GC_FGD(1): 确定前景
            # cv2.GC_PR_BGD(2): 可能背景, cv2.GC_PR_FGD(3): 可能前景
            gc_mask = np.zeros(roi_color.shape[:2], np.uint8)
            
            # 默认整个扩充过的区域都是“可能背景”
            gc_mask[:] = cv2.GC_PR_BGD
            
            # 将深度学习网络高亮的地方标记为“可能前景”
            # 这一步是让 AI 和传统图像处理结合的灵魂
            gc_mask[roi_dl_mask > 0] = cv2.GC_PR_FGD
            
            # 分配 GrabCut 所需的内存 (必须是 1x65 的 float64 数组)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            
            # 4. 执行 GrabCut 算法细化边缘 (迭代 2 次即可，保证实时性)
            try:
                cv2.grabCut(roi_color, gc_mask, None, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_MASK)
            except cv2.error:
                # 极端情况下（如 ROI 太小或颜色单一导致 GMM 崩溃），退回到原始 DL Mask
                pass 
                
            # 5. 提取细化后的精准 Mask
            # 找出被 GrabCut 判定为“确定前景”或“可能前景”的像素
            refined_anomaly_pixels = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), True, False)
            
            # 6. 仅对精准的像素点上色并叠加
            overlay_roi = overlay[y1:y2, x1:x2]
            overlay_roi[refined_anomaly_pixels] = mask_color
            overlay[y1:y2, x1:x2] = overlay_roi
            
            # 绘制 ID 与日志记录
            cv2.putText(display_frame, f"ID:{t['id']}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            with open("alert_log.txt", "a") as f:
                f.write(f"Alert: FrameID:{frame_count}, TargetID:{t['id']}, Center:({x+w//2},{y+h//2}), Area:{t['area']}\n")
        
        # 将涂好颜色的 overlay 图层与原图进行半透明混合
        cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
        
        # 3. 计算并绘制 FPS
        frame_count += 1
        elapsed = time.time() - fps_start_time
        fps = frame_count / elapsed
        print(frame_count, f"FPS: {fps:.1f}")
        # cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 4. 显示与录像
        # cv2.imshow("Dynamic Small Target Anomaly Detection", display_frame)
        
        if video_writer is not None:
            video_writer.write(display_frame)
            
        q_anomaly_maps.task_done()
        
        # 5. 无 GUI 环境下 cv2.waitKey 会失效，将其注释掉。
        # 如果处理的是视频文件，视频读完后代码会自动结束。
        # 如果是实时流，请在终端按 Ctrl+C 来强制终止程序。
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     print("\n收到退出指令，正在通知底层工作线程停止...")
        #     stop_event.set()

    print("正在等待工作线程安全退出 (可能需要几秒钟)...")
    t_prod.join()
    t_gpu.join()
    
    if video_writer is not None:
        video_writer.release()
        
    # cv2.destroyAllWindows()
    print("引擎已彻底关闭。")

if __name__ == "__main__":
    # 如果单独运行此文件，默认使用 0 号摄像头进行测试，不保存视频
    main_consumer_pipeline(video_source=0, weights_path='checkpoints/best_student.pth', save_video=False)