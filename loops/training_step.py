import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
# import clip
import warnings
from tqdm import tqdm
from collections import defaultdict
from json import dumps
from termcolor import colored
import clip
from utils.utils import (
    create_logits,
    gen_label,
    convert_models_to_fp32,
    AverageMeter,
    compute_accept_mask,
    compute_acc,
)
from utils.attributes import (
    compute_clustering_accuracy,
    extract_cluster_attributes,
    match_attributes,
    extract_class_attributes,
)

# 训练步骤函数
def training_step(
    model,
    loader,
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
    prompt_learner, #cococo_prompt_learner(新增)
    thresholds=None,
    samples_per_pseudo_labels=None,
):
    # 忽略 ResourceWarning
    warnings.simplefilter("ignore", ResourceWarning)

    # 定义训练过程中需要记录的指标
    train_total_loss_source = AverageMeter()  # 源域总损失
    train_loss_video_source = AverageMeter()  # 源域视频损失
    train_loss_text_source = AverageMeter()  # 源域文本损失
    train_total_loss_target = AverageMeter()  # 目标域总损失
    train_loss_video_target = AverageMeter()  # 目标域视频损失
    train_loss_text_target = AverageMeter()  # 目标域文本损失
    train_total_loss_domain = AverageMeter()  # 域损失
    train_loss_video_domain = AverageMeter()  # 域视频损失
    train_loss_text_domain = AverageMeter()  # 域文本损失
    pseudo_label_acc = AverageMeter()  # 伪标签准确率
    entropy_loss_target = AverageMeter()  # 目标域熵损失
    consistency_loss_target = AverageMeter()  # 一致性损失
    accepted_pseudo_labels = 0  # 被接受的伪标签数量
    total_pseudo_labels = 0  # 总伪标签数量

    # 将模型设置为训练模式
    for m in model:
        model[m].train()

    # 存储伪标签的置信度
    new_samples_per_pseudo_labels = defaultdict(list)

    # 聚类处理
    matching_results = None
    # 检查配置文件中是否使用"autolabel"聚类方法
    if config.open_set.method == "autolabel":
        # 使用指定的聚类方法对目标域数据进行聚类
        clustering_results = clustering_method(model, loader["target"], config, device)
        #clustering_results["labels"]表示每个数据点被分配到的聚类标签,表示数据点属于第几个聚类。
        #clustering_results["ids"] 表示每个数据点的唯一标识符（ID）。这些ID通常是原始数据集中的索引或者唯一标记，用于将聚类结果与原始数据对应起来。
        #clustering_results["true_labels"]表示每个数据点的真实标签。这些标签是数据点的实际分类信息，通常在聚类任务中用于评估聚类的效果。例如，在一个手写数字数据集上，真实标签可能是数字 0 到 9，表示每个数据点对应的实际类别。

        # 提取聚类结果中的目标域每个簇的属性，并获取每个簇的真实标签
        (
            target_attributes_per_cluster,  # 每个簇的目标域属性
            true_labels_per_cluster,  # 每个簇的真实标签
        ) = extract_cluster_attributes(clustering_results, config, epoch, run_id)

        # 输出正在提取目标域簇的属性
        print(colored("Exctracting attributes...", "green"))
        
        # 提取源域每个类的属性
        source_attributes_per_class = extract_class_attributes(config)  # 提取源域类的属性
        
        # 输出属性提取完成，开始进行属性匹配
        print(colored("Attributes extracted. Matching attributes...", "green"))
        
        # 使用源域类属性和目标域簇属性进行匹配
        matching_results = match_attributes(
            source_attributes_per_class, target_attributes_per_cluster, config
        )
        
        # 计算聚类的准确率
        clustering_accuracy = compute_clustering_accuracy(
            matching_results["matches_per_cluster_label"],  # 匹配的簇标签
            clustering_results["labels"],  # 聚类结果的标签
            clustering_results["true_labels"],  # 真实标签
            config,  # 配置文件
        )
        
        # 输出聚类准确率
        print("Clustering accuracy: {}".format(clustering_accuracy))
        
        # 记录聚类准确率到日志系统
        if config.logging.wandb:  # 如果启用了 wandb 日志
            wandb.log({"clustering_accuracy": clustering_accuracy})  # 记录到 wandb
        if config.logging.neptune:  # 如果启用了 Neptune 日志
            run["clustering_accuracy"].log(clustering_accuracy)  # 记录到 Neptune
        if config.logging.comet:  # 如果启用了 Comet 日志
            experiment.log_metric(
                "clustering_accuracy", clustering_accuracy, step=epoch
            )  # 记录到 Comet
        
        # 输出属性匹配完成
        print(colored("Attributes matched", "green"))

        # 如果配置文件要求详细输出聚类匹配结果，则输出匹配结果
        if config.logging.verbose:
            # 输出匹配结果的详细信息
            print(
                dumps(
                    matching_results["matches_per_cluster_label"],  # 匹配的每个簇标签
                    indent=2,  # 格式化输出
                )
            )


    # 打印当前训练的Epoch信息
    print("[Training] - Epoch {}".format(epoch + 1))
    # 遍历训练集的每个batch
    for batch_idx, batch in enumerate(tqdm(loader["train"])):
        # 根据不同的数据集配置获取源域和目标域数据
        if "ek" in config.data.dataset:
            (
                video_source,
                y_source,
                id_source,
                video_target,
                y_target,
                _,
                _,
                _,
                id_target,
                target_index,
            ) = batch
        else:
            (
                video_source,
                y_source,
                id_source,
                video_target,
                y_target,
                _,
                id_target,
                target_index,
                skeleton_source, 
                skeleton_target
            ) = batch

        # 如果solver类型不是"monitor"，则每10个batch更新一次scheduler
        if config.solver.type != "monitor":
            if (batch_idx + 1) == 1 or (batch_idx + 1) % 10 == 0:
                warnings.simplefilter("ignore")
                scheduler.step(epoch + batch_idx / len(loader))
                warnings.resetwarnings()

        # 清零梯度
        optimizers["main_optimizer"].zero_grad()

        # --------------------------- 处理源域数据 --------------------------- #

        # 调整源域视频输入形状并获取尺寸
        video_source = video_source.view(
            (-1, config.data.num_segments, 3) + video_source.size()[-2:]
        )
        b, t, c, h, w = video_source.size()

        # 生成源域的文本数据
        text_id = np.random.randint(prompts["num_text_aug"], size=len(y_source))
        texts_source = torch.stack(
            [prompts["text_dict"][j][i, :] for i, j in zip(y_source, text_id)]
        )

        # 将数据移到GPU设备
        video_source = video_source.to(device).view(-1, c, h, w)
        texts_source = texts_source.to(device)
        skeleton_source = skeleton_source.to(device).float()    

        # 生成视频和文本的嵌入表示
        image_features = model["video_model"](video_source)
        image_features = image_features.view(b, t, -1).float()  
        # print(f"image_features size: {image_features.size()}") #torch.Size([6, 8, 512])  
        # print(f"video_embedding_test1 size: {video_embedding_test1.size()}") #torch.Size([6, 8, 512])
        video_embedding_test1 = model['HierarchicalCrossModalFusion_model'](image_features, skeleton_source) 

        # 进行时间维度聚合，假设使用平均池化
        video_embedding = video_embedding_test1.mean(dim=1).float()

        learnable_prompts = model["prompt_learner"](video_embedding_test1)
        tokenized_prompts = model["prompt_learner"].tokenized_prompts
        cococo_text_features = model["text_model"](learnable_prompts,tokenized_prompts,True).float()# 现在应为 [B, n_class, dim]

        # print(f"text_embedding: {text_embedding.shape}")             #torch.Size([6, 512])
        # print(f"cococo_text_features: {cococo_text_features.shape}") #torch.Size([6, 6, 512])

        # logit_scale 是一个参数，用于缩放视频和文本的相似度分数（logits）
        logit_scale = model["full"].logit_scale.exp().float()

        y_source = y_source.to(device)

        video_embedding = F.normalize(video_embedding, dim=-1)
        text_features = F.normalize(cococo_text_features, dim=-1)
        logits_list = []
        batch_size, num_classes = text_features.shape[0], text_features.shape[1]
        for i in range(batch_size):
            # 每个视频计算与所有类别的相似度
            logit = logit_scale * video_embedding[i] @ text_features[i].permute(1,0)  # [1, num_classes]
            logits_list.append(logit)
        logits_per_video = torch.stack(logits_list)  # [B, C]
        # y_source应为类别index而非one-hot
        loss_video_source = F.cross_entropy(logits_per_video.contiguous(), y_source)

        total_loss_source = loss_video_source  # 单侧损失
        total_loss = config.loss.source.weight * total_loss_source

        # 更新统计指标
        # assert logits_per_video.size(0) == logits_per_text.size(0)
        train_total_loss_source.update(
            total_loss_source.item(), logits_per_video.size(0)
        )
        train_loss_video_source.update(
            loss_video_source.item(), logits_per_video.size(0)
        )
        # train_loss_text_source.update(loss_text_source.item(), logits_per_video.size(0))

        # --------------------------- 处理目标域数据 --------------------------- #
        # 如果目标域的损失权重存在（不为零），则执行以下代码
        if config.loss.target.weight:

            # 调整目标域视频输入形状并获取尺寸
            video_target = video_target.view(
                (-1, config.data.num_segments, 3) + video_target.size()[-2:]
            )
            b, t, c, h, w = video_target.size()

            extended_classes = prompts["classes"]  # 扩展后的类
            text_dict_target = prompts["text_dict_target"]  # 目标域文本字典

            # 根据open_set方法选择处理方式
            if config.open_set.method in ["autolabel", "oracle"]:
                if config.open_set.method == "autolabel":
                    if config.attributes.clustering_method in ["kmeans", "hodge"]:
                        extended_classes_names = [
                            c for _, c in prompts["classes_names"]
                        ] + matching_results["open_set_labels"]
                    else:
                        raise ValueError(
                            "Clustering algorithm {} not recognized!".format(
                                config.attributes.clustering_method
                            )
                        )
                else:
                    extended_classes_names = [
                        c for _, c in prompts["classes_names"]
                    ] + prompts["open_set_labels"]
                # 重新构造目标域的文本字典
                text_dict_target = {}
                for i, txt in enumerate(prompts["text_aug"]):
                    text_dict_target[i] = torch.cat(
                        [clip.tokenize(txt.format(c)) for c in extended_classes_names]
                    )
                extended_classes = torch.cat([v for _, v in text_dict_target.items()])
                # print(f"extended_classes_names: {extended_classes_names}")#extended_classes_names: ['climb', 'fencing', 'golf', 'kick ball', 'pullup', 'punch', 'grass and tree and man', 'green and ball and man', 'table and person and man', 'referee and ring and man', 'bike and fence and person', 'door and mirror and man', 'horse and fence and person', 'dog and person and man', 'wall and person and man', 'basketball and ball and male', 'net and people and ball']
                model["prompt_learner"]._build_prompts(extended_classes_names)
                # 将扩展类写入文件
                with open(
                    "{}_extended_classes.txt".format(run_id), "a"
                ) as extended_classes_file:
                    extended_classes_file.write(
                        "\n{}: {}\n".format(epoch, extended_classes_names)
                    )

            # 将目标域数据移到GPU
            video_target = video_target.to(device).view(-1, c, h, w)
            image_features = model["video_model"](video_target)
            image_features = image_features.view(b, t, -1)
            image_features = image_features.to(device).float()
            skeleton_target = skeleton_target.to(device).float()
            video_embedding = model['HierarchicalCrossModalFusion_model'](image_features, skeleton_target) 
            

            learnable_prompts = model["prompt_learner"](video_embedding)
            tokenized_prompts = model["prompt_learner"].tokenized_prompts
            cococo_text_features = model["text_model"](learnable_prompts,tokenized_prompts,True)# 现在应为 [B, n_class, dim]
            cococo_text_features = cococo_text_features.float()
            video_embedding = video_embedding.mean(dim=1).float()
            # 计算伪标签和接受掩码
            mask, accepted, pseudo_y_target, values = compute_accept_mask(
                model,
                video_embedding,
                extended_classes,
                prompts,
                device,
                config,
                thresholds,
                samples_per_pseudo_labels,
                target_index,
            )

            # 收集目标样本及其置信度
            for i in range(b):
                label = pseudo_y_target[i]  # 获取伪标签
                entry = (target_index[i].item(), values[i].item())  # 获取目标样本的索引和置信度值
                new_samples_per_pseudo_labels[label.item()].append(entry)  # 按标签存储

            # 统计总伪标签数量
            total_pseudo_labels += b

            # 处理接受的伪标签
            if accepted:

                # 统计接受的伪标签数量
                accepted_pseudo_labels += accepted
                y_target = y_target.to(device)  # 将目标标签转移到GPU
                mask = mask.to(device)  # 将掩码移到GPU
                pseudo_y_target = pseudo_y_target.to(device)  # 将伪标签移到GPU
                pseudo_label_accuracy = compute_acc(
                    y_target[mask].to(device), pseudo_y_target[mask]
                )  # 计算伪标签准确率
                pseudo_label_acc.update(
                    pseudo_label_accuracy, pseudo_y_target[mask].size(0)
                )  # 更新伪标签准确率

                # 获取目标标签，若使用地面真实标签则直接使用目标标签，否则使用伪标签
                if config.loss.target.use_gt:
                    target_label = y_target
                else:
                    target_label = pseudo_y_target
                # print(f"pseudo_y_target: {pseudo_y_target}") #pseudo_y_target: tensor([4, 8, 4, 1, 0, 7], device='cuda:0')
                # print(f"y_target: {y_target}")               #y_target: tensor([4, 6, 4, 0, 0, 3], device='cuda:0')
                # print(f"target_label: {target_label}")       #target_label: tensor([4, 8, 4, 1, 0, 7], device='cuda:0')
                text_id = np.random.randint(prompts["num_text_aug"], size=b)  # 随机生成文本索引
                texts_target = torch.stack(
                    [text_dict_target[j][i, :] for i, j in zip(target_label, text_id)]
                )  # 获取目标文本
                # 将数据移到GPU
                mask = mask.to(device)
                texts_target = texts_target.to(device)
                texts_target = texts_target[mask].to(device)  # 只保留mask中为True的样本
                # print(f"texts_target[mask]: {texts_target.size()}") #texts_target[mask]: torch.Size([6, 77])
                # 生成视频和文本的嵌入表示
                video_embedding = video_embedding[mask]  # 过滤掉不需要的样本

                if config.loss.target.weight:

                    # logit_scale 是一个参数，用于缩放视频和文本的相似度分数（logits）
                    logit_scale = model["full"].logit_scale.exp()
                    logit_scale = logit_scale.float()

                    video_embedding = F.normalize(video_embedding, dim=-1)
                    text_features = F.normalize(cococo_text_features, dim=-1)
                    logits_list = []
                    batch_size, num_classes = text_features.shape[0], text_features.shape[1]
                    for i in range(batch_size):
                        # 每个视频计算与所有类别的相似度
                        logit = logit_scale * video_embedding[i] @ text_features[i].permute(1,0)  # [1, num_classes]
                        logits_list.append(logit)
                    logits_per_video = torch.stack(logits_list)  # [B, C]

                    loss_video_target = F.cross_entropy(logits_per_video.contiguous(), y_target)

                    total_loss_target = loss_video_target  # 单侧损失

                    total_loss = config.loss.target.weight * total_loss_target

                    # 更新统计指标
                    train_total_loss_target.update(
                        total_loss_target.item(), logits_per_video.size(0)
                    )  # 更新目标域总损失
                    train_loss_video_target.update(
                        loss_video_target.item(), logits_per_video.size(0)
                    )  # 更新目标域视频损失
                    # train_loss_text_target.update(
                    #     loss_text_target.item(), logits_per_video.size(0)
                    # )  # 更新目标域文本损失

            # 反向传播计算梯度
            total_loss.backward()

            # 优化步骤
            if device == "cpu":
                optimizers["main_optimizer"].step()  # 如果是CPU，直接更新
            else:
                convert_models_to_fp32(model["full"])  # 转换模型到FP32
                optimizers["main_optimizer"].step()  # 更新优化器
                clip.model.convert_weights(model["full"])  # 转换CLIP模型的权重

        torch.cuda.empty_cache()

    # 计算接受的伪标签的百分比
    percentage_accepted_pseudo_labels = 0
    if config.loss.target.weight:
        percentage_accepted_pseudo_labels = accepted_pseudo_labels / total_pseudo_labels  # 计算接受伪标签的比例
        percentage_accepted_pseudo_labels *= 100  # 转换为百分比

    # 记录日志到Comet
    if config.logging.comet:
        experiment.log_metric(
            "train_total_loss_source", train_total_loss_source.avg, step=epoch
        )  # 记录源域总损失
        experiment.log_metric(
            "train_loss_video_source", train_total_loss_source.avg, step=epoch
        )  # 记录源域视频损失
        experiment.log_metric(
            "train_loss_text_source", train_loss_text_source.avg, step=epoch
        )  # 记录源域文本损失
        experiment.log_metric(
            "train_total_loss_target", train_total_loss_target.avg, step=epoch
        )  # 记录目标域总损失
        experiment.log_metric(
            "train_loss_video_target", train_loss_video_target.avg, step=epoch
        )  # 记录目标域视频损失
        experiment.log_metric(
            "train_loss_text_target", train_loss_text_target.avg, step=epoch
        )  # 记录目标域文本损失
        experiment.log_metric(
            "entropy_loss_target", entropy_loss_target.avg, step=epoch
        )  # 记录目标域熵损失
        experiment.log_metric("pseudo_labels_acc", pseudo_label_acc.avg, step=epoch)  # 记录伪标签准确率
        experiment.log_metric(
            "accepted_pseudo_labels(%)", percentage_accepted_pseudo_labels, step=epoch
        )  # 记录接受伪标签的比例
        if config.open_set.method == "attributes":
            experiment.log_metric(
                "open_set_labels",
                len(matching_results["open_set_labels"]),
                step=epoch,
            )  # 记录开集标签数量

    # 重置各类统计指标
    train_total_loss_source.reset()
    train_loss_video_source.reset()
    train_loss_text_source.reset()
    train_total_loss_target.reset()
    train_loss_video_target.reset()
    train_loss_text_target.reset()
    train_total_loss_domain.reset()
    train_loss_video_domain.reset()
    train_loss_text_domain.reset()
    entropy_loss_target.reset()
    consistency_loss_target.reset()
    pseudo_label_acc.reset()

    # 按照置信度排序伪标签
    for label in new_samples_per_pseudo_labels:
        new_samples_per_pseudo_labels[label] = [
            index
            for index, _ in sorted(
                new_samples_per_pseudo_labels[label], key=lambda x: x[1], reverse=True
            )
        ]

    # 构造返回结果
    res = {"samples_per_pseudo_labels": new_samples_per_pseudo_labels}
    if config.open_set.method == "autolabel":
        res["open_set_labels"] = matching_results["open_set_labels"]  # 如果使用自动标签，返回开集标签
    elif config.open_set.method == "oracle":
        res["open_set_labels"] = prompts["open_set_labels"]  # 如果使用oracle，返回开集标签

    return res
