import cv2
import os
from PIL import Image
import numpy as np

# 输入视频的根文件夹路径
input_root_folder = r"C:\Users\傅裕\Desktop\通过视频提示传输有效的视频无监督域适应实现鲁棒的人类动作识别\autolabel-main\dataset\hmdb-ucf\hmdb-ucf"
# 输出文件夹路径
output_root_folder = r"C:\Users\傅裕\Desktop\通过视频提示传输有效的视频无监督域适应实现鲁棒的人类动作识别\autolabel-main\dataset\hmdb-ucf\RGB-feature"

# 遍历输入文件夹中的所有子文件夹
for root, dirs, files in os.walk(input_root_folder):
    # 忽略文件夹路径的根目录
    if root == input_root_folder:
        continue
    
    # 根据输入的文件夹结构，创建对应的输出文件夹
    relative_path = os.path.relpath(root, input_root_folder)
    output_folder = os.path.join(output_root_folder, relative_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历当前文件夹中的所有视频文件
    for video_file in files:
        if video_file.endswith(('.avi', '.mp4', '.mov')):  # 根据需要添加更多视频格式
            video_path = os.path.join(root, video_file)
            
            # 创建以视频文件名命名的子文件夹
            video_folder_name = os.path.splitext(video_file)[0]
            video_output_folder = os.path.join(output_folder, video_folder_name)
            if not os.path.exists(video_output_folder):
                os.makedirs(video_output_folder)

            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"无法打开视频文件: {video_path}")
                continue

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"视频 {video_file} 读取结束或读取失败")
                    break

                # 保存每一帧图像
                frame_filename = os.path.join(video_output_folder, f"frame_{frame_count:04d}.jpg")
                try:
                    # 使用 OpenCV 保存帧
                    success = cv2.imwrite(frame_filename, frame)
                    if not success:
                        # 使用 PIL 替代保存
                        print(f"OpenCV 保存失败，尝试使用 PIL...")
                        frame_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        frame_img.save(frame_filename)
                    print(f"成功保存帧 {frame_count:04d} 到 {frame_filename}")
                except Exception as e:
                    print(f"保存帧 {frame_count:04d} 时发生错误：{e}")

                frame_count += 1

            cap.release()

print(f"视频转帧完成，所有视频的帧已保存到对应文件夹中。")
