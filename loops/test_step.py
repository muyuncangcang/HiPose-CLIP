import torch
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
import numpy as np
import warnings
import clip
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils.utils import LabelsManager
from PIL import UnidentifiedImageError, Image
from termcolor import colored
from json import load


def test_step(
    model,  # 模型字典，包含了视频模型、文本模型、融合模型等
    loader,  # 数据加载器（通常是一个字典，其中包含了训练、验证、测试数据）
    config,  # 配置参数
    prompts,  # 包含了类别和文本增强的提示信息
    device,  # 设备（例如 "cuda" 或 "cpu"）
    classes_names,  # 类别名称
    epoch,  # 当前的训练轮次
    run_id,  # 运行ID
    prompt_learner, #cococo_prompt_learner(新增)
    training_results=None,  # 训练过程中产生的结果（可选）
    sanity_check=False,  # 是否进行快速的测试检查
    run=None,  # wandb运行对象（可选）
    experiment=None,  # 实验对象（用于日志记录，comet 或 wandb）
):

    # 忽略 ResourceWarning 警告
    warnings.simplefilter("ignore", ResourceWarning)

    # 将所有模型设置为评估模式
    for m in model:
        model[m].eval()

    # 初始化计数器
    num = 0  # 样本数量
    corr_1 = 0  # top-1准确率
    corr_5 = 0  # top-5准确率

    num_classes = len(classes_names)  # 类别数量
    # print(classes_names)

    loader = loader["val"]  # 获取验证集
    if sanity_check:
        print(colored("Sanity check...", "green"))  # 如果是sanity check，打印绿色提示信息
    else:
        print("[Testing] - Epoch {}".format(epoch + 1))  # 否则打印测试信息
        loader = tqdm(loader)  # 使用tqdm显示加载进度

    with torch.no_grad():  # 禁用梯度计算，节省内存

        classes = prompts["classes"]  # 从提示信息中获取类名
        new_classes = [c for _, c in classes_names]

        # 文本编码
        if not sanity_check:
            if config.open_set.method in ["autolabel", "oracle"]:  # 如果使用了autolabel或oracle方法
                text_dict = {}
                for i, txt in enumerate(prompts["text_aug"]):  # 遍历每个文本增强
                    text_dict[i] = torch.cat(
                        [clip.tokenize(txt.format(c)) for _, c in classes_names]  # 为每个类别生成文本token
                    )
                    if len(training_results["open_set_labels"]):  # 如果有开放集标签
                        text_dict[i] = torch.cat(
                            (
                                text_dict[i],
                                torch.cat(
                                    [
                                        clip.tokenize(open_set_label)
                                        for open_set_label in training_results[
                                            "open_set_labels"
                                        ]
                                    ]
                                ),
                            )
                        )
                classes = torch.cat([v for _, v in text_dict.items()])  # 合并所有文本特征
                extended_classes_names = [c for _, c in classes_names] + training_results["open_set_labels"]  # 扩展类别名称
                model["prompt_learner"]._build_prompts(extended_classes_names)

        text_inputs = classes.to(device)  # 将文本输入发送到设备上
        # text_features = model["text_model"](text_inputs)  # 通过文本模型提取文本特征
        
        # print(text_features.shape) #torch.Size([96, 512])
        # 用于保存预测标签和真实标签
        pred = []
        gt = []
        ids = []
        tps = []

        # 每个类别的正确预测数和类别实例数
        correct_per_class = [0 for _ in range(num_classes + 1)]
        instances_per_class = [0 for _ in range(num_classes + 1)]

        attributes_file_path = "utils/attributes/{}_test_{}_vilt.json".format(
            config.data.target_dataset, config.attributes.n_blanks
        )

        # 如果是olympics数据集，更新路径
        if "olympics" in config.data.dataset:
            attributes_file_path = attributes_file_path.replace(
                "_test_", "_test_ucfolympics_"
            )
        if config.data.clean_ek:  # 如果需要清洗EK数据
            attribute_file_path = attributes_file_path.replace("_test_", "_test_clean_")

        # 遍历验证集中的批次
        for batch_idx, batch in enumerate(loader):
            video, y, video_id, skeleton_source = batch
            # 在数据预处理阶段将未知类标签映射为 num_classes
            y = torch.where(y >= num_classes, num_classes, y)
            
            # 计算视频特征
            video = video.view((-1, config.data.num_segments, 3) + video.size()[-2:])  # 调整视频张量形状
            b, t, c, h, w = video.size()  # 获取batch大小、时间段数、通道数、高度和宽度
            y = y.to(device)  # 将标签发送到设备上
            video_input = video.to(device).view(-1, c, h, w)  # 将视频输入发送到设备上
            video_features = model["video_model"](video_input).view(b, t, -1)  # 通过视频模型提取视频特征
            # print(video_features.shape)    #torch.Size([6, 8, 512])
            # print(learnable_prompts.shape) #torch.Size([6, 6, 77, 512])
            # print(tokenized_prompts.shape) #torch.Size([6, 77])
            image_features = video_features / video_features.norm(dim=-1, keepdim=True)  # L2归一化
            skeleton_source = skeleton_source.to(device).float()
            image_features = image_features.to(device).float()
            fusion_video_features = model['HierarchicalCrossModalFusion_model'](image_features, skeleton_source) 
            learnable_prompts = model["prompt_learner"](fusion_video_features)
            tokenized_prompts = model["prompt_learner"].tokenized_prompts
            cococo_text_features = model["text_model"](learnable_prompts,tokenized_prompts,True)# 现在应为 [B, n_class, dim]
            fusion_video_features = fusion_video_features.mean(dim=1).float()
            fusion_video_features = fusion_video_features.float()
            # 处理开放集协议（open set protocol）
            if config.open_set.method == "zoc":  # 使用ZOC方法时
                assert b == 1, "Val batch size must be 1!"  # 验证集batch大小必须为1
                text_dict = {}
                for i, txt in enumerate(prompts["text_aug"]):
                    text_dict[i] = torch.cat(
                        [clip.tokenize(txt.format(c)) for _, c in classes_names]
                    )
                    with open(attributes_file_path, "r") as attributes_file:
                        open_set_labels = load(attributes_file)[str(video_id.item())]  # 加载开放集标签
                        print(f"open_set_labels:{open_set_labels}")
                        text_dict[i] = torch.cat(
                            (
                                text_dict[i],
                                torch.cat(
                                    [
                                        clip.tokenize(open_set_label)
                                        for open_set_label in open_set_labels
                                    ]
                                ),
                            )
                        )
                        classes = torch.cat([v for _, v in text_dict.items()])
                        text_inputs = classes.to(device)
                        text_features = model["text_model"](text_inputs)

            # 归一化视频和文本特征
            fusion_video_features = F.normalize(fusion_video_features, p=2, dim=-1)
            cococo_text_features = F.normalize(cococo_text_features, p=2, dim=-1)
            
            # 计算相似度
            cococo_text_features = cococo_text_features.float()

            # 对每个视频样本独立处理
            logit_scale = model["full"].logit_scale.exp()
            logit_scale = logit_scale.float()
            similarities = []
            for i in range(b):
                # 获取单个视频特征 [1, D]
                video_feat = fusion_video_features[i].unsqueeze(0)  # [1, 512]
                # 获取对应样本的文本特征 [C, D]
                text_feat = cococo_text_features[i]  # [6, 512]
                # 计算相似度矩阵
                sim = logit_scale * video_feat @ text_feat.T  # [1, C]
                similarities.append(sim)
            similarity = torch.cat(similarities, dim=0)  # [B, C]

            # 计算top-1和top-5的准确率
            values_1, indices_1 = similarity.topk(1, dim=-1)
            if "olympics" not in config.data.dataset:
                values_5, indices_5 = similarity.topk(5, dim=-1)
            num += b  # 增加总样本数

            # 开放集协议的处理
            if config.open_set.method in ["autolabel", "oracle", "zoc"]:
                predicted_indices = indices_1.clone()
                indices_1[indices_1 >= num_classes] = num_classes
            elif config.open_set.method == "osvm":
                indices_1[values_1 < config.open_set.threshold] = num_classes
            else:
                raise ValueError("Open set method not recognized!")  # 未知的开放集方法

            # 跟踪性能
            for i in range(b):
                label = y[i]
                if indices_1[i] == label:
                    corr_1 += 1  # 如果预测正确，top-1准确率加1
                if "olympics" not in config.data.dataset:
                    if y[i] in indices_5[i]:
                        corr_5 += 1  # 如果标签在top-5中，top-5准确率加1
                predicted_label = indices_1[i]
                instances_per_class[label] += 1  # 每个类别的实例数加1
                if label.item() == predicted_label.item():
                    correct_per_class[label] += 1  # 如果预测正确，该类别的正确数加1
            pred.extend([i[0] for i in indices_1.cpu().tolist()])  # 将预测标签加入列表
            gt.extend(list(y.cpu().tolist()))  # 将真实标签加入列表
            ids.extend(list(video_id.cpu().tolist()))  # 将视频ID加入列表

            # 如果是sanity check，提前结束循环
            if sanity_check:
                if batch_idx >= config.general.sanity_check_steps:
                    break

            torch.cuda.empty_cache()

    # 计算top-1准确率
    acc1 = float(corr_1) / num * 100
    if "olympics" not in config.data.dataset:
        acc5 = float(corr_5) / num * 100  # 计算top-5准确率

    # 开放集的准确率和H-score
    h_score = 0.0
    if not sanity_check:
        accuracy_per_class = np.array(correct_per_class) / np.array(instances_per_class)  # 计算每个类别的准确率
        closed_accuracy = accuracy_per_class[:num_classes].mean()  # 计算封闭集的平均准确率
        open_accuracy = accuracy_per_class[-1]  # 计算开放集的准确率
        h_score = (
            2 * closed_accuracy * open_accuracy / (closed_accuracy + open_accuracy)
        )  # 计算H-score

    # 日志记录
    if not sanity_check:

        labels_manager = LabelsManager(config)

        # 计算并保存混淆矩阵
        labels = [i[1] for i in classes_names]
        labels.append("UNK")
        cm = confusion_matrix(
            labels_manager.convert(gt),
            labels_manager.convert(pred),
            labels=labels,
        )
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(xticks_rotation="vertical")
        plt.savefig("cm.png")  # 保存混淆矩阵图像
        plt.close()

        if config.logging.comet:  # 如果启用了comet日志
            experiment.log_metric("validation_accuracy", acc1, step=epoch)  # 记录top-1准确率
            if "olympics" not in config.data.dataset:
                experiment.log_metric("acc5", acc5, step=epoch)  # 记录top-5准确率
            experiment.log_confusion_matrix(
                gt,
                pred,
                labels=labels,
            )
            experiment.log_image(Image.open("cm.png"), name="cm_epoch={}".format(epoch))  # 记录混淆矩阵图片
            experiment.log_metric("closed_accuracy", closed_accuracy, step=epoch)  # 记录封闭集准确率
            experiment.log_metric("open_accuracy", open_accuracy, step=epoch)  # 记录开放集准确率
            experiment.log_metric("h_score", h_score, step=epoch)  # 记录H-score

        # 打印并记录日志
        if "olympics" not in config.data.dataset:
            line = (
                "Epoch: [{}/{}]:\n  VAL ACCURACY @1: {}\n  VAL ACCURACY @5: {}".format(
                    epoch + 1, config.solver.epochs, acc1, acc5
                )
            )
        else:
            line = "Epoch: [{}/{}]:\n  VAL ACCURACY @1: {}".format(
                epoch + 1, config.solver.epochs, acc1
            )
        line += (
            "\n  CLOSED ACCURACY: {}\n  OPEN ACCURACY: {}\n  H-SCORE: {}".format(
                closed_accuracy, open_accuracy, h_score
            )
        )
        print(line)  # 打印当前结果
        with open(config.log_file_path, "a") as logfile:
            logfile.write("{}\n\n".format(line))  # 将结果写入日志文件

    res = h_score  # 返回H-score

    return res
