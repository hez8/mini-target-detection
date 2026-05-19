# prepare_templates.py
import os
import cv2
import glob
import numpy as np

def rotate_image_with_alpha(image, angle):
    """对带有透明通道的图像进行无损防裁切旋转"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 计算旋转后的新边界尺寸，防止图像边角被切断
    abs_cos, abs_sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    M[0, 2] += new_w / 2 - center[0]
    M[1, 2] += new_h / 2 - center[1]
    
    # 使用全透明背景填充空白区域
    rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                             flags=cv2.INTER_LINEAR, 
                             borderMode=cv2.BORDER_CONSTANT, 
                             borderValue=(0, 0, 0, 0))
    return rotated

def main():
    input_dir = 'data/anomaly_templates'
    output_dir = 'data/anomaly_templates_rotated' # 新的图库文件夹
    os.makedirs(output_dir, exist_ok=True)
    
    template_paths = glob.glob(os.path.join(input_dir, "*.png"))
    
    if not template_paths:
        print(f"[-] 在 {input_dir} 中找不到 PNG 图片！请确保你放入了带有透明通道的抠图。")
        return
        
    print(f"[*] 找到 {len(template_paths)} 个原始异物模板。开始执行矩阵旋转膨胀...")
    
    total_generated = 0
    for path in template_paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        if img is None or img.shape[2] != 4:
            print(f"[警告] 跳过无效或无透明通道的图片: {path}")
            continue
            
        base_name = os.path.splitext(os.path.basename(path))[0]
        
        # 每 10 度旋转一次，一张图能生成 36 张不同角度的变体
        for angle in range(0, 360, 10):
            rotated_img = rotate_image_with_alpha(img, angle)
            out_path = os.path.join(output_dir, f"{base_name}_rot{angle}.png")
            cv2.imwrite(out_path, rotated_img)
            total_generated += 1
            
    print(f"[+] 空间换时间完成！共生成了 {total_generated} 张预旋转模板图。")
    print(f"[*] 请去 {output_dir} 文件夹下查看生成结果。")

if __name__ == "__main__":
    main()