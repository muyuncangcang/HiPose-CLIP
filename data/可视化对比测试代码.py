import cv2
import matplotlib.pyplot as plt
from pathlib import Path
# from data.dataset import VideoDataset
from PIL import Image
import numpy as np
import re
import torch.utils.data as data
# from .data_utils import natural_keys
import datetime
import pickle
import cv2
import torch
import torch.utils.data as data
import os
import os.path
import numpy as np
from numpy.random import randint
from PIL import Image
from random import Random
from torch.utils.data import Dataset, DataLoader
# from .data_utils import natural_keys

from os import listdir
from os.path import join

class VideoDataset(data.Dataset):
    def __init__(
        self,
        dataset_input,
        root,
        num_segments=1,
        new_length=1,
        frame_tmpl="{:05d}.jpg",
        transform=None,
        random_shift=True,
        test_mode=False,
        index_bias=1,
        epic_kitchens=False,
        slurm=False,
        alderaan=False,
        hpc=False,
        open_set=False,
        return_paths=False,
    ):

        self.root = root
        self.num_segments = num_segments
        self.seg_length = new_length
        self.frame_tmpl = frame_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop = False
        self.index_bias = index_bias
        self.epic_kitchens = epic_kitchens
        self.slurm = slurm
        self.alderaan = alderaan
        self.hpc = hpc
        self.open_set = open_set
        self.return_paths = return_paths

        if self.index_bias is None:
            if self.frame_tmpl == "frame{:d}.jpg":
                self.index_bias = 0
            else:
                self.index_bias = 1

        self.classes = []
        self.video_list = []

        self.parse_file(dataset_input)

        self.initialized = False

    def parse_file(self, dataset_input):
        with open(dataset_input, "r") as file_list:
            i = 0
            for line in file_list:
                split_line = line.split()
                path = split_line[0]
                # path = join(self.root, path)
                # ddd = path.split("/")
                # path = ddd[1]
                path = self.root+"/"+path
                if self.epic_kitchens:
                    start_frame = int(split_line[1])
                    stop_frame = int(split_line[2])
                    label = int(split_line[3])
                    self.video_list.append((path, start_frame, stop_frame, label, i))
                    kitchen = path.split("/")[-1]
                    if kitchen not in self.ek_videos:
                        kitchen_videos = self.find_frames(path)
                        kitchen_videos.sort(key=natural_keys)
                        self.ek_videos[kitchen] = kitchen_videos
                else:
                    label = int(split_line[1])
                    self.video_list.append((path, label, i))
                i += 1

    def _load_image(self, directory, idx):
        image = Image.open(os.path.join(directory, self.frame_tmpl.format(idx))).convert("RGB")
        # Convert to numpy array and then to float32
        image = np.array(image, dtype=np.float32)
        return image

    @property
    def total_length(self):
        return self.num_segments * self.seg_length

    def find_frames(self, video):
        frames = [join(video, f) for f in listdir(video) if self.is_img(f)]
        return frames

    # checks if input is image
    def is_img(self, f):
        return str(f).lower().endswith("jpg") or str(f).lower().endswith("jpeg")


    def _calculate_frame_differences(self, frame_paths):
        """计算每帧与总体均值的差异"""
        features = []
        for fp in frame_paths:
            img = Image.open(fp).convert('L')  # 转为灰度图
            img_array = np.array(img)
            features.append(img_array.mean())
        
        total_mean = np.mean(features)
        return [abs(f - total_mean) for f in features]

    def _sample_indices(self, video):
        """关键帧采样方法"""
        frame_paths = self.find_frames(video)
        if not frame_paths:
            return np.array([], dtype=int)
        
        frame_paths.sort(key=natural_keys)
        differences = self._calculate_frame_differences(frame_paths)
        
        N = self.num_segments * self.seg_length
        num_frames = len(frame_paths)
        
        # 获取差异最大的N个帧的索引
        sorted_indices = np.argsort(differences)[::-1]  # 降序排列
        
        if N > num_frames:
            if self.loop:
                selected_indices = np.mod(np.arange(N), num_frames)
            else:
                selected_indices = np.concatenate([np.arange(num_frames), np.zeros(N - num_frames, dtype=int)])
        else:
            selected_indices = sorted_indices[:N]
        
        # 添加index_bias并确保不越界
        selected_indices = np.clip(selected_indices, 0, len(frame_paths)-1) + self.index_bias
        return selected_indices.astype(int)

    def _get_val_indices(self, video):

        num_segments = self.num_segments

        length = len(self.find_frames(video))
        
        if num_segments == 1:
            return np.array([length // 2], dtype=int) + self.index_bias

        if length <= self.total_length:
            if self.loop:
                return np.mod(np.arange(self.total_length), length) + self.index_bias
            return (
                np.array(
                    [i * length // self.total_length for i in range(self.total_length)],
                    dtype=int,
                )
                + self.index_bias
            )
        offset = (length / num_segments - self.seg_length) / 2.0
        return (
            np.array(
                [
                    i * length / num_segments + offset + j
                    for i in range(num_segments)
                    for j in range(self.seg_length)
                ],
                dtype=int,
            )
            + self.index_bias
        )

    def get_video_2(self, label1):
        label2 = label1
        video2 = None

        while label2 == label1:
            index = randint(0, self.__len__())
            if self.epic_kitchens:
                video, start_frame, stop_frame, label2 = self.video_list[index]
                video2 = {
                    "video": video,
                    "start_frame": start_frame,
                    "stop_frame": stop_frame,
                }
            else:
                video2, label2 = self.video_list[index]

        assert video2 is not None
        assert label2 != label1

        return video2, label2

    def __getitem__(self, index):

        video, label, video_id = self.video_list[index]
        # print(video_id)
        segment_indices = (
            self._sample_indices(video)
            if self.random_shift #  random_shift: True
            else self._get_val_indices(video)
        )

        return self.get(video, label, segment_indices, video_id)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

    def get(self, video, label, indices, video_id):
        images = list()

        # find frames
        path = video
        frame_paths = self.find_frames(video)
        frame_paths.sort(key=natural_keys)

        for i, seg_ind in enumerate(indices):
            p = int(seg_ind) - 1
            try:
                seg_imgs = [Image.open(frame_paths[p]).convert("RGB")]
            except OSError:
                print('ERROR: Could not read image "{}"'.format(video))
                print("invalid indices: {}".format(indices))
                raise
            images.extend(seg_imgs)

        process_data = self.transform(images)

        return process_data, label, path, video_id

    def __len__(self):
        return len(self.video_list)


class VideoDataset2(data.Dataset):
    def __init__(
        self,
        dataset_input,
        root,
        num_segments=1,
        new_length=1,
        frame_tmpl="{:05d}.jpg",
        transform=None,
        random_shift=True,
        test_mode=False,
        index_bias=1,
        epic_kitchens=False,
        slurm=False,
        alderaan=False,
        hpc=False,
        open_set=False,
        return_paths=False,
    ):

        self.root = root
        self.num_segments = num_segments
        self.seg_length = new_length
        self.frame_tmpl = frame_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop = False
        self.index_bias = index_bias
        self.epic_kitchens = epic_kitchens
        self.slurm = slurm
        self.alderaan = alderaan
        self.hpc = hpc
        self.open_set = open_set
        self.return_paths = return_paths

        if self.index_bias is None:
            if self.frame_tmpl == "frame{:d}.jpg":
                self.index_bias = 0
            else:
                self.index_bias = 1

        self.classes = []
        self.video_list = []

        self.parse_file(dataset_input)

        self.initialized = False

    def parse_file(self, dataset_input):
        with open(dataset_input, "r") as file_list:
            i = 0
            for line in file_list:
                split_line = line.split()
                path = split_line[0]
                # path = join(self.root, path)
                # ddd = path.split("/")
                # path = ddd[1]
                path = self.root+"/"+path
                if self.epic_kitchens:
                    start_frame = int(split_line[1])
                    stop_frame = int(split_line[2])
                    label = int(split_line[3])
                    self.video_list.append((path, start_frame, stop_frame, label, i))
                    kitchen = path.split("/")[-1]
                    if kitchen not in self.ek_videos:
                        kitchen_videos = self.find_frames(path)
                        kitchen_videos.sort(key=natural_keys)
                        self.ek_videos[kitchen] = kitchen_videos
                else:
                    label = int(split_line[1])
                    self.video_list.append((path, label, i))
                i += 1

    def _load_image(self, directory, idx):
        image = Image.open(os.path.join(directory, self.frame_tmpl.format(idx))).convert("RGB")
        # Convert to numpy array and then to float32
        image = np.array(image, dtype=np.float32)
        return image

    @property
    def total_length(self):
        return self.num_segments * self.seg_length

    def find_frames(self, video):
        frames = [join(video, f) for f in listdir(video) if self.is_img(f)]
        return frames

    # checks if input is image
    def is_img(self, f):
        return str(f).lower().endswith("jpg") or str(f).lower().endswith("jpeg")

    def _sample_indices(self, video):
        offsets = list()

        num_segments = self.num_segments

        length = len(self.find_frames(video))

        ticks = [i * length // num_segments for i in range(num_segments + 1)]

        for i in range(num_segments):
            tick_len = ticks[i + 1] - ticks[i]
            tick = ticks[i]
            if tick_len >= self.seg_length:
                tick += randint(tick_len - self.seg_length + 1)
            offsets.extend([j for j in range(tick, tick + self.seg_length)])
        return np.array(offsets) + self.index_bias

    def _get_val_indices(self, video):

        num_segments = self.num_segments

        length = len(self.find_frames(video))
        
        if num_segments == 1:
            return np.array([length // 2], dtype=np.int) + self.index_bias

        if length <= self.total_length:
            if self.loop:
                return np.mod(np.arange(self.total_length), length) + self.index_bias
            return (
                np.array(
                    [i * length // self.total_length for i in range(self.total_length)],
                    dtype=np.int32,
                )
                + self.index_bias
            )
        offset = (length / num_segments - self.seg_length) / 2.0
        return (
            np.array(
                [
                    i * length / num_segments + offset + j
                    for i in range(num_segments)
                    for j in range(self.seg_length)
                ],
                dtype=int,
            )
            + self.index_bias
        )

    def get_video_2(self, label1):
        label2 = label1
        video2 = None

        while label2 == label1:
            index = randint(0, self.__len__())
            if self.epic_kitchens:
                video, start_frame, stop_frame, label2 = self.video_list[index]
                video2 = {
                    "video": video,
                    "start_frame": start_frame,
                    "stop_frame": stop_frame,
                }
            else:
                video2, label2 = self.video_list[index]

        assert video2 is not None
        assert label2 != label1

        return video2, label2

    def __getitem__(self, index):

        video, label, video_id = self.video_list[index]
        # print(video_id)
        segment_indices = (
            self._sample_indices(video)
            if self.random_shift #  random_shift: True
            else self._get_val_indices(video)
        )

        return self.get(video, label, segment_indices, video_id)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

    def get(self, video, label, indices, video_id):
        images = list()

        # find frames
        if self.epic_kitchens:
            path = video["video"]
            kitchen = path.split("/")[-1]
            frame_paths = self.ek_videos[kitchen]
            frame_paths = frame_paths[video["start_frame"] : video["stop_frame"]]
        else:
            path = video
            frame_paths = self.find_frames(video)
            frame_paths.sort(key=natural_keys)

        for i, seg_ind in enumerate(indices):
            p = int(seg_ind) - 1
            try:
                seg_imgs = [Image.open(frame_paths[p]).convert("RGB")]
            except OSError:
                print('ERROR: Could not read image "{}"'.format(video))
                print("invalid indices: {}".format(indices))
                raise
            images.extend(seg_imgs)

        process_data = self.transform(images)

        if self.return_paths:
            if self.epic_kitchens:
                return (
                    process_data,
                    label,
                    path,
                    video["start_frame"],
                    video["stop_frame"],
                    video_id,
                )
            return process_data, label, path, video_id

        return process_data, label, path, video_id

    def __len__(self):
        return len(self.video_list)



def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", str(text))]


def visualize_comparison(video_path, num_segments=3, seg_length=5):
    # 初始化两种数据集
    orig_dataset = VideoDataset2(
        dataset_input="/opt/data/private/txt/hmdb-ucf/hmdb_test_target.txt",
        root=str(Path(video_path).parent),
        num_segments=num_segments,
        new_length=seg_length,
        random_shift=True,
        test_mode=False,
        epic_kitchens=False,
        frame_tmpl="img_{:05d}.jpg"  # 需与实际文件名格式一致
    )
    
    keyframe_dataset = VideoDataset(
        dataset_input="/opt/data/private/txt/hmdb-ucf/hmdb_test_target.txt",
        root=str(Path(video_path).parent),
        num_segments=num_segments,
        new_length=seg_length,
        random_shift=False,
        test_mode=True,
        epic_kitchens=False,
        frame_tmpl="img_{:05d}.jpg"  # 保持与文件实际命名一致
    )

    # 获取帧路径并检查有效性
    frame_dir = Path(video_path)
    if not frame_dir.exists():
        raise FileNotFoundError(f"视频帧目录不存在: {frame_dir}")
    
    frame_paths = sorted([f for f in frame_dir.glob("*.jpg")], key=natural_keys)
    if not frame_paths:
        raise ValueError("目录中未找到任何帧图像")

    # 获取采样结果并验证索引
    try:
        orig_indices = orig_dataset._sample_indices(str(frame_dir))
        keyframe_indices = keyframe_dataset._sample_indices(str(frame_dir))
    except Exception as e:
        print(f"采样时发生错误: {str(e)}")
        return

    # 创建对比可视化
    plt.figure(figsize=(20, 10))
    
    # 绘制原始采样
    for i, idx in enumerate(orig_indices[:5]):
        plt.subplot(2, 5, i+1)
        if idx <= len(frame_paths):
            img = Image.open(frame_paths[idx-1])
            plt.imshow(img)
        plt.axis('off')
        plt.title(f"Original\nFrame {idx}")

    # 绘制关键帧采样
    for i, idx in enumerate(keyframe_indices[:5]):
        plt.subplot(2, 5, i+6)
        if idx <= len(frame_paths):
            img = Image.open(frame_paths[idx-1])
            plt.imshow(img)
        plt.axis('off')
        plt.title(f"Keyframe\nFrame {idx}")

    plt.tight_layout()
    plt.savefig("frame_comparison.png")  # 保存为图片避免依赖GUI
    plt.close()

    # 生成对比视频（修复版）
    output_path = "comparison.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用更通用的编解码器
    out = None
    
    try:
        out = cv2.VideoWriter(output_path, fourcc, 5.0, (1280, 480))
        
        for idx in range(max(len(orig_indices), len(keyframe_indices))):
            frame_pair = np.zeros((480, 1280, 3), dtype=np.uint8)
            
            # 原始帧处理
            if idx < len(orig_indices):
                orig_path = frame_paths[orig_indices[idx]-1]
                orig_frame = cv2.imread(str(orig_path))
                if orig_frame is not None:
                    orig_frame = cv2.resize(orig_frame, (640, 480))
                    frame_pair[:, :640] = orig_frame
            
            # 关键帧处理
            if idx < len(keyframe_indices):
                key_path = frame_paths[keyframe_indices[idx]-1]
                key_frame = cv2.imread(str(key_path))
                if key_frame is not None:
                    key_frame = cv2.resize(key_frame, (640, 480))
                    frame_pair[:, 640:] = key_frame
            
            # 添加文字标注
            cv2.putText(frame_pair, f"Original Sampling (Frame {orig_indices[idx] if idx<len(orig_indices) else 'N/A'})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(frame_pair, f"Keyframe Sampling (Frame {keyframe_indices[idx] if idx<len(keyframe_indices) else 'N/A'})", 
                       (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            out.write(frame_pair)
            
    except Exception as e:
        print(f"视频生成失败: {str(e)}")
    finally:
        if out is not None:
            out.release()
            print(f"对比视频已保存至: {Path(output_path).resolve()}")

# 使用示例（需替换实际路径）
if __name__ == "__main__":
    # test_video_path = "/opt/data/private/dataset/hmdb-ucf/RGB-feature"  # 包含帧图像的目录
    test_video_path = "/opt/data/private/dataset/hmdb-ucf/RGB-feature/climb/(HQ)_Rock_Climbing_-_Free_Solo_Speed_Climb_-_Dan_Osman_climb_f_cm_np1_ba_med_0"
    if Path(test_video_path).exists():
        visualize_comparison(test_video_path)
    else:
        print(f"测试路径不存在: {test_video_path}")