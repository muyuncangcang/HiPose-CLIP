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
from .data_utils import natural_keys

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


class VideoDatasetSourceAndTarget:
    def __init__(self, source_dataset, target_dataset):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def __len__(self):
        return max([len(self.source_dataset), len(self.target_dataset)])

    def __getitem__(self, index):
        source_index = index % len(self.source_dataset)
        source_data = self.source_dataset[source_index]

        target_index = index % len(self.target_dataset)
        target_data = self.target_dataset[target_index]
        return (*source_data, *target_data, target_index)


class CombinedDataset(Dataset):
    def __init__(self, video_dataset, skeleton_train, skeleton_val):
        self.video_dataset = video_dataset
        self.skeleton_train = skeleton_train
        self.skeleton_val = skeleton_val
        self._build_mapping()

    def save_mapping(self):
        # 创建保存文件夹，如果不存在的话
        folder_path = '/root/ddddddd/autolabel-main/data/Mapping'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 使用当前时间戳来命名文件，避免覆盖
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"mapping_{timestamp}.txt"
        file_path = os.path.join(folder_path, file_name)

        # 将字典转换为字符串并保存为txt文件
        with open(file_path, 'w') as f:
            for video_id, idx in self.mapping.items():
                f.write(f"{video_id}: {idx}\n")

        print(f"Mapping saved to: {file_path}")

    def _build_mapping(self):
        self.mapping = {}
        # 映射训练集骨骼数据
        for idx in range(len(self.skeleton_train)):
            skeleton_item = self.skeleton_train[idx]
            skeleton_video_id = os.path.splitext(os.path.basename(skeleton_item['video_id']))[0]
            self.mapping[skeleton_video_id] = ('train', idx)
        # 映射验证集骨骼数据（避免键冲突）
        for idx in range(len(self.skeleton_val)):
            skeleton_item = self.skeleton_val[idx]
            skeleton_video_id = os.path.splitext(os.path.basename(skeleton_item['video_id']))[0]
            if skeleton_video_id not in self.mapping:  # 训练集优先
                self.mapping[skeleton_video_id] = ('val', idx)

    def __len__(self):
        return len(self.video_dataset)

    def __getitem__(self, index):
        video_item = self.video_dataset[index]
    
        # 解析VideoDatasetSourceAndTarget的特殊结构
        if isinstance(self.video_dataset, VideoDatasetSourceAndTarget):
            # 源数据可能是3元素，目标数据是4元素的结构
            source_data = video_item[:-5]  # 假设末尾4个元素是目标数据相关
            target_data = video_item[-5:]
            # print(video_item)
            video_data_source, y_source, path_source, id_source = source_data[0], source_data[1], source_data[2], source_data[3]
            video_data_target, y_target, path_target, id_target, target_index = target_data
            
            video_filename_source = os.path.basename(str(path_source))
            video_id_clean_source = os.path.splitext(video_filename_source)[0]
            video_filename_target = os.path.basename(str(path_target))
            video_id_clean_target = os.path.splitext(video_filename_target)[0]

            # 查找骨骼数据来源
            source, idx = self.mapping.get(video_id_clean_source, (None, -1))
            skeleton_source = self.skeleton_train[idx]['skeleton']
            # 查找骨骼数据来源
            source, idx = self.mapping.get(video_id_clean_target, (None, -1))
            skeleton_target = self.skeleton_val[idx]['skeleton']
            
            return (
                video_data_source, y_source, id_source, 
                video_data_target, y_target, path_target, id_target, target_index,
                skeleton_source, skeleton_target
            )
        else:   
            video_data = video_item[0]
            label = video_item[1]
            path = video_item[2] if len(video_item)>3 else None
            video_id = video_item[-1]  # 总取最后一个元素作为video_id

            video_filename = os.path.basename(str(path))
            video_id_clean = os.path.splitext(video_filename)[0]
            
            # 查找骨骼数据来源
            source, idx = self.mapping.get(video_id_clean, (None, -1))
            if source == 'train':
                skeleton_data = self.skeleton_train[idx]['skeleton']
            elif source == 'val':
                skeleton_data = self.skeleton_val[idx]['skeleton']
            else:
                skeleton_data = torch.zeros(200, 17, 2)  # 默认空骨架
            
            return {
                "video": video_data,
                "skeleton": skeleton_data,
                "label": label,
                "video_id": video_id
            }

    def _get_skeleton(self, idx):
        if idx == -1:
            return torch.zeros(200,17,2)  # 空骨骼数据
        return self.skeleton_dataset[idx]['skeleton']
    
# 数据加载模块
class SkeletonDataset(Dataset):
    def __init__(self, pkl_path, clip_feat_dim=512):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # 智能判断数据格式
        if isinstance(data, dict) and 'annotations' in data:
            # 格式1：数据存储在字典的annotations字段
            self.annotations = data['annotations']
            # print(f'Detected dict format with {len(self.annotations)} annotations')
        elif isinstance(data, list):
            # 格式2：数据直接是annotations列表
            self.annotations = data
            # print(f'Detected direct list format with {len(self.annotations)} annotations')
        else:
            raise ValueError('Unsupported data format!')
        
        self.clip_feat_dim = clip_feat_dim
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # 骨骼数据处理
        keypoints = ann['keypoint']  # [M, T, V, C]
        keypoint_score = ann['keypoint_score']  # [M, T, V]
        
        # 选择主要人物（取置信度最高的）
        if keypoints.shape[0] > 1:
            scores = keypoint_score.mean(axis=(1,2))  # [M]
            main_person = scores.argmax()
            keypoints = keypoints[main_person]  # [T, V, C]
        elif keypoints.shape[0] == 1:
            keypoints = keypoints[0]
        else:
            raise RuntimeError(f"Logic error: keypoints shape {keypoints.shape}")
        
        # 归一化处理
        keypoints = keypoints - keypoints[..., :1, :]  # 以第一个关节为基准
        
        # # CLIP特征
        # clip_feat = self.clip_features[ann['frame_dir']]  # [T, 512]
        
        return {
            'skeleton': torch.FloatTensor(keypoints),  # [T, V, C]
            # 'clip_feat': torch.FloatTensor(clip_feat), # [T, 512]
            'label': ann['label'],
            'video_id': ann['frame_dir']  # 添加视频ID字段
        }
    
from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    # 判断是否是复合结构（包含源和目标）
    if isinstance(batch[0], tuple):  # 来自VideoDatasetSourceAndTarget的原始结构
        # 解包源和目标数据
        (video_source, y_source, id_source, 
         video_target, y_target, path_target, id_target, target_index,
         skeleton_source, skeleton_target) = zip(*batch)
        
        # 处理视频数据
        video_source = torch.stack(video_source)
        video_target = torch.stack(video_target)
        
        # 处理骨骼数据
        skeleton_source = pad_sequence(skeleton_source, batch_first=True)
        skeleton_target = pad_sequence(skeleton_target, batch_first=True)
        
        return (
            video_source, torch.tensor(y_source), torch.tensor(id_source),
            video_target, torch.tensor(y_target), path_target, torch.tensor(id_target), torch.tensor(target_index),
            skeleton_source, skeleton_target
        )
    else:  # 来自单个数据集的结构
        videos = torch.stack([item['video'] for item in batch])
        skeletons = pad_sequence([item['skeleton'] for item in batch], batch_first=True)
        labels = torch.tensor([item['label'] for item in batch])
        video_ids = torch.tensor([item['video_id'] for item in batch])
        return (videos, labels, video_ids, skeletons)
