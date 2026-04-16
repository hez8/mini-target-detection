# data_preprocess.py
import os
import cv2
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def process_single_video(video_path, output_dir, target_fps=5):
    """
    处理单个视频：按指定 FPS 抽帧并保存
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return f"[错误] 无法打开视频: {video_name}"

    # 获取视频原始信息
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 防止读取到异常的 FPS (如 0)
    if orig_fps <= 0 or orig_fps != orig_fps:
        orig_fps = 30.0 

    # 计算抽帧步长 (例如原视频30帧，目标5帧，则每6帧抽一张)
    frame_interval = max(1, int(round(orig_fps / target_fps)))
    
    count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 按步长抽帧
        if count % frame_interval == 0:
            # 命名规范：视频名_frame_00001.jpg
            out_name = f"{video_name}_frame_{saved_count:05d}.jpg"
            out_path = os.path.join(output_dir, out_name)
            
            # 保存图片 (为了保证训练质量，建议保存为 95% 质量的 JPEG)
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1
            
        count += 1
        
    cap.release()
    return f"[成功] {video_name} -> 原始帧:{total_frames}, 抽出图片:{saved_count}张"

def main():
    # ================= 配置区 =================
    # 存放原始视频的文件夹路径
    INPUT_DIR = "data/raw_videos" 
    
    # 抽帧后图片存放的文件夹路径 (必须与 default_config.py 中的 TRAIN_DATA_DIR 一致)
    OUTPUT_DIR = "data/train_frames" 
    
    # 目标抽帧率 (建议 5-10 之间。云台转得慢就设为5，转得快就设为10)
    TARGET_FPS = 5 
    
    # 支持的视频格式
    VIDEO_EXTENSIONS = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    # ==========================================

    # 1. 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2. 收集所有视频文件
    video_paths = []
    for ext in VIDEO_EXTENSIONS:
        # 支持大小写后缀匹配
        video_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
        video_paths.extend(glob.glob(os.path.join(INPUT_DIR, ext.upper())))
        
    if not video_paths:
        print(f"在 {INPUT_DIR} 目录下没有找到任何视频文件！请检查路径。")
        return

    print(f"=== 开始视频抽帧预处理 ===")
    print(f"找到 {len(video_paths)} 个视频，目标帧率设定为 {TARGET_FPS} FPS")
    print(f"输出目录: {OUTPUT_DIR}")
    print("--------------------------------------------------")

    start_time = time.time()
    
    # 3. 启动多进程池并行处理
    # max_workers 默认使用系统的 CPU 核心数
    with ProcessPoolExecutor() as executor:
        # 提交所有任务
        futures = [executor.submit(process_single_video, vp, OUTPUT_DIR, TARGET_FPS) for vp in video_paths]
        
        # 实时打印进度
        for future in as_completed(futures):
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"[异常] 任务执行失败: {e}")

    elapsed_time = time.time() - start_time
    total_images = len(os.listdir(OUTPUT_DIR))
    
    print("--------------------------------------------------")
    print(f"=== 预处理完成 ===")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"输出文件夹中现共有图片: {total_images} 张")
    print("现在你可以运行 python main.py --mode train 开始无监督训练了！")

if __name__ == "__main__":
    main()