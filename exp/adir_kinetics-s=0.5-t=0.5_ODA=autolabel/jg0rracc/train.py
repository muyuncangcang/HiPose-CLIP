# 引入必要的库和模块
import multiprocessing
from comet_ml import Experiment  # 用于与 Comet.ml 平台交互，记录实验过程
from data.data_utils import get_transforms, get_classes, get_open_set_classes  # 数据处理相关工具函数
from data.dataloader import get_dataloaders  # 数据加载器
# from modules.cnn import CNNModule
# from modules.CNN import CNNDLGA
# from modules.prompts import St_gcn, visual_prompt, text_prompt  # 提示生成模块
from modules.prompts import CoCoOpPromptLearner, CrossModalFusion, HierarchicalMMAFusion, PoseEncoderGCN, visual_prompt, text_prompt  # 提示生成模块
from modules.clip_modules import ImageCLIP, TextCLIP  # CLIP 模型的图像和文本部分
from modules.kll_loss import KLLoss  # 自定义损失函数 KLLoss
from utils.solver import _optimizer, _lr_scheduler  # 优化器和学习率调度器
from utils.utils import epoch_saving, best_saving, process_run_name  # 辅助工具：保存模型、处理运行名称
from utils.attributes import hodeg_clustering, k_means  # K-Means 聚类方法5

from loops.test_step import test_step  # 测试循环的实现
from omegaconf import DictConfig, OmegaConf, open_dict  # 配置文件解析工具
from os.path import join  # 文件路径拼接工具
from pathlib import Path  # 文件路径处理工具
from shutil import copy  # 文件拷贝工具
from termcolor import colored  # 用于终端输出彩色文本

import warnings  # 警告模块
import wandb  # 用于与 Weights & Biases 平台交互
import torch  # PyTorch 深度学习框架
import torch.nn as nn
import clip  # OpenAI 的 CLIP 模型
import hydra  # 用于配置管理和实验跟踪
import sys  # 系统交互工具
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter("ignore", ResourceWarning)
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib import rcParams
# 设置字体为默认字体
# rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# 设置随机种子以确保实验的可重复性
def seed_everything(seed: int):
    import random, os
    import numpy as np

    random.seed(seed)  # Python 原生随机数生成器的种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置 Python 哈希种子
    np.random.seed(seed)  # NumPy 随机数种子
    torch.manual_seed(seed)  # PyTorch CPU 随机数种子
    torch.cuda.manual_seed(seed)  # PyTorch GPU 随机数种子
    torch.backends.cudnn.deterministic = True  # 确保 CuDNN 的确定性
    torch.backends.cudnn.benchmark = True  # 加速 CuDNN 的计算

# 使用 Hydra 管理配置文件，设置主入口函数
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):

    warnings.simplefilter("ignore", ResourceWarning)  # 忽略资源警告

    seed_everything(config.seed)  # 设置随机种子

    from loops.training_step import training_step  # 延迟导入训练循环

    # 处理运行名称并打印
    run_name, run_id, run_name_no_id = process_run_name(config)
    print(colored("RUN ID: {}".format(run_id), "green"))
    print(colored("RUN NAME: {}".format(run_name), "green"))

    # 初始化 WandB（Weights & Biases）日志记录
    if config.logging.wandb:
        wandb.init(
            project=config.logging.project_name,
            name=run_name,
            config=dict(config),
        )
    run = None

    # 初始化 Comet.ml 日志记录
    experiment = None
    if config.logging.comet:
        experiment = Experiment(
            api_key="wIf8mPYXi87PpERUtIBhKOX3c",
            project_name=config.logging.project_name,
            workspace="gzaraunitn",
        )
        experiment.set_name(name=run_name)
        experiment.log_parameters(dict(config))  # 记录配置参数
        experiment.log_parameter("command", " ".join(sys.argv))  # 记录运行命令
        if config.logging.tag:
            experiment.add_tag(config.logging.tag)

    # 创建实验的工作目录
    working_dir = join("./exp", run_name_no_id, run_id)
    Path(working_dir).mkdir(parents=True, exist_ok=True)

    # 保存配置文件和主脚本到实验目录
    with open_dict(config):
        config.log_file_path = join(working_dir, "log.txt")
    copy("/opt/data/private/3-30第二次hodge聚类编写/configs/config.yaml", working_dir)
    copy("/opt/data/private/3-30第二次hodge聚类编写/train.py", working_dir)

    # 检查设备是否支持 CUDA
    device = torch.device('cuda:0') if torch.cuda.is_available() else "cpu"

    # 加载 CLIP 模型
    model, clip_state_dict = clip.load(
        config.network.arch,# ViT-B/32 ViT-B/16
        device=device,
        jit=False,#False
        tsm=config.network.tsm,
        T=config.data.num_segments,
        dropout=config.network.dropout,
        emb_dropout=config.network.emb_dropout,
        pretrain=config.network.init,
        joint=config.network.joint,
    )

    # 创建文本提示
    classes_names = get_classes(config)
    # CoCoOp_classes_names = 
    prompts = text_prompt(classes_names)
    prompts["classes_names"] = classes_names

    # 获取数据预处理转换
    training_transforms, test_transforms = get_transforms(config)

    model = model.to(device)
    # 初始化 CLIP 模块
    video_model = ImageCLIP(model)  # 视频模型
    fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)  # 融合模型9
    # text_model = TextCLIP(model)  # 文本模型
    text_model = TextCLIP(model)  # 从配置启用提示
    PoseEncoder_model = PoseEncoderGCN(input_dim=2)
    CrossModalFusion_model = CrossModalFusion()
    prompt_learner = CoCoOpPromptLearner(config, classes_names, model)
    HierarchicalMMAFusion_model = HierarchicalMMAFusion(num_layers=3)

    video_model = video_model.to(device)
    fusion_model = fusion_model.to(device)
    text_model = text_model.to(device)
    PoseEncoder_model = PoseEncoder_model.to(device).float()
    CrossModalFusion_model = CrossModalFusion_model.to(device).float()
    HierarchicalMMAFusion_model = HierarchicalMMAFusion_model.to(device).float()
    prompt_learner.to(device)

    # 如果启用 WandB，监控模型
    if config.logging.wandb:
        wandb.watch(model)
        wandb.watch(fusion_model)

    # 获取数据加载器
    dataloaders = get_dataloaders(
        config=config,
        training_transforms=training_transforms,
        test_transforms=test_transforms,
    )

    # 根据设备类型调整模型权重
    if device == "cpu":
        text_model.float()
        video_model.float()
    else:
        clip.model.convert_weights(text_model)
        clip.model.convert_weights(video_model)
        
    model = model.to(device)

    # 定义损失函数
    loss_video = nn.CrossEntropyLoss()
    loss_text = nn.CrossEntropyLoss()

    # 初始化训练相关变量
    start_epoch = config.solver.start_epoch
    best_score = 0.0

    if config.open_set.method == "oracle":
        print(colored("Fetching open-set class names...", "green"))
        prompts["open_set_labels"] = get_open_set_classes(config)# 返回开放集的类别名称列表
        print(colored("Open-set class names fetched", "green"))

    # 加载预训练模型权重
    if config.network.pretrained_model:
        print(colored("Loading pretrained weights...", "green"))
        full_model_params = torch.load(config.network.pretrained_model, map_location="cpu")["model_state_dict"]
        model.load_state_dict(full_model_params)

    # 选择聚类方法
    if config.attributes.clustering_method == "kmeans":
        clustering_method = k_means
    elif config.attributes.clustering_method == "hodge":
        clustering_method = hodeg_clustering
    else:
        raise ValueError("Clustering method {} not recognized!".format(config.attributes.clustering_method))

    # 分组模型
    models = {
        "full": model,
        "video_model": video_model,
        "text_model": text_model,
        "fusion_model": fusion_model,
        "pose_encoder": PoseEncoder_model,  # 新增
        "fusion": CrossModalFusion_model,   # 新增
        "prompt_learner": prompt_learner,   # 新增
        "HierarchicalCrossModalFusion_model": HierarchicalMMAFusion_model
    }

    # 如果加载了预训练权重，初始化融合模型
    if config.network.pretrained_model:
        print(colored("Loading fusion model pretrained weights...", "green"))
        fusion_model_params = torch.load(
            config.network.pretrained_model, map_location="cpu"
        )["fusion_model_state_dict"]
        models["fusion_model"].load_state_dict(fusion_model_params)
        print(colored("Weights loaded", "green"))

    # 分组损失函数
    loss = {"loss_video": loss_video, "loss_text": loss_text}

    # 定义优化器和学习率调度器
    optimizers = _optimizer(config, models)
    scheduler = None
    if config.solver.type != "monitor":
        scheduler = _lr_scheduler(config, optimizers["main_optimizer"])

    samples_per_pseudo_labels = None

    # 主训练循环
    for epoch in range(start_epoch, config.solver.epochs):

        if epoch == 0 and config.general.sanity_check:
            _ = test_step(
                models,
                dataloaders,
                config,
                prompts,
                device,
                classes_names,
                epoch,
                run_id,
                prompt_learner,
                sanity_check=True,
            )
            print(colored("Sanity check ok!\n", "green"))

        training_results = training_step(
            models,
            dataloaders,
            clustering_method,
            optimizers,
            scheduler,
            config,
            prompts,
            device,
            loss,
            epoch,
            run,
            experiment,
            run_id,
            prompt_learner,
            samples_per_pseudo_labels,
        )

        samples_per_pseudo_labels = training_results["samples_per_pseudo_labels"]

        score = test_step(
            models,
            dataloaders,
            config,
            prompts,
            device,
            classes_names,
            epoch,
            run_id,
            prompt_learner,
            training_results=training_results,
            run=run,
            experiment=experiment,
        )

        is_best = score > best_score
        best_score = max(score, best_score)

        line = "Current/Best: {}/{}\n".format(score, best_score)
        print(line)
        with open(config.log_file_path, "a") as logfile:
            logfile.write("{}\n\n".format(line))

        # log best score
        if config.logging.comet:
            experiment.log_metric("best_score", best_score, step=epoch)

        if config.logging.save:
            print("Saving...")
            filename = "{}/last_model.pt".format(working_dir)

            epoch_saving(
                epoch, model, fusion_model, optimizers["main_optimizer"], filename
            )
            if is_best:
                best_saving(
                    working_dir,
                    epoch,
                    model,
                    fusion_model,
                    optimizers["main_optimizer"],
                )
            print("Saved\n")

    if config.logging.comet:
        experiment.end()


if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn', force=True)
    main()
