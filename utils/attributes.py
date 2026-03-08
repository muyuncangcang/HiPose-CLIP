from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import NearestNeighbors
from skimage.feature import hog
import cv2
from skimage.color import rgb2gray
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd
import numpy as np
# from numpy import ctypeslib
# ctypeslib.load_library('npymath', np.get_include()) 
# import hdbscan
from tqdm import tqdm
from utils.utils import LabelsManager
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from json import load
from random import choice
from yellowbrick.cluster import KElbowVisualizer
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
import scipy.sparse.csgraph as csgraph
import cupy as cp
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import lsqr
from sklearn.neighbors import NearestNeighbors

def hodeg_clustering(model, loader, config, device):
    res = {}
    
    # 模型评估模式
    for m in model.values():
        m.eval()

    # 特征提取
    labels_manager = LabelsManager(config)
    feats, ids, true_labels = [], [], []
    
    with torch.no_grad():
        print("Spectral Clustering: Loading training set")
        for batch_idx, (video_target, y_target, video_id, skeleton) in enumerate(tqdm(loader)):
            video_target = video_target.view(
                (-1, config.data.num_segments, 3) + video_target.size()[-2:]
            )
            b, t, c, h, w = video_target.size()
            
            video_embedding = model["video_model"](
                video_target.to(device).view(-1, c, h, w)
            ).view(b, t, -1)
            
            fused_embedding = model["fusion_model"](video_embedding)

            for i in range(b):
                feats.append(fused_embedding[i].flatten().cpu().numpy())
                ids.append(video_id[i].item())
                true_labels.append(labels_manager.convert_single_label(y_target[i].item()))

    # 数据标准化
    feats = StandardScaler().fit_transform(feats)
    
    # 构建KNN图（保持稀疏格式）
    nbrs = NearestNeighbors(n_neighbors=config.spectral.get('n_neighbors', 10))
    nbrs.fit(feats)
    adj_matrix = nbrs.kneighbors_graph(feats)
    symmetric_adj = 0.5 * (adj_matrix + adj_matrix.T)  # 高效对称化

    def HHD(adj_sparse):
        """改进的Hodge分解实现"""
        num_nodes = adj_sparse.shape[0]
        
        # 构建度矩阵
        degree_values = adj_sparse.sum(axis=1).A1.astype(np.float64)
        degree_matrix = sp.diags(degree_values, format='csr')
        
        # 构建组合拉普拉斯矩阵
        Lv = degree_matrix - adj_sparse.astype(np.float64)
        
        # 增加正则化项确保矩阵正定
        Lv += 1e-6 * sp.eye(num_nodes, format='csr')
        
        # 改进的线性系统求解
        result = lsqr(Lv, degree_values, atol=1e-5, btol=1e-5, iter_lim=1000)
        x = result[0]
        
        return x

    print("Computing Hodge decomposition...")
    node_potentials = HHD(symmetric_adj)
    
    # 构建相似度矩阵（稀疏优化版）
    print("Building Hodge similarity matrix...")
    n_samples = len(feats)
    similarity_matrix = adj_matrix.astype(float).tolil()
    
    gamma = config.spectral.get('gamma', 1.0)
    rows, cols = symmetric_adj.nonzero()
    for i, j in zip(rows, cols):
        if i < j:
            diff = node_potentials[i] - node_potentials[j]
            similarity = np.exp(-gamma * (diff ** 2))
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    
    # 转换为适合谱聚类的格式
    similarity_matrix = similarity_matrix.tocsr()

    # 自动化确定聚类数目（优化特征值计算）
    def auto_detect_k(similarity_matrix, max_k=15):
        from scipy.sparse.linalg import eigsh
        laplacian = csgraph.laplacian(similarity_matrix, normed=True)
        eigenvalues = eigsh(laplacian, k=max_k, which='SM')[0]
        eigenvalues.sort()
        
        eigengaps = np.diff(eigenvalues[:max_k])
        return np.argmax(eigengaps) + 1

    # 确定聚类数目
    if config.attributes.auto_k:
        print("Automatically detecting optimal K...")
        k = auto_detect_k(similarity_matrix)
        print(f"Auto detected K = {k}")
    else:
        k = config.attributes.k_clustering

    # 谱聚类执行
    print(f"Performing spectral clustering with k={k}...")
    cluster_model = SpectralClustering(
        n_clusters=k,
        affinity='precomputed',
        random_state=config.seed,
        assign_labels='cluster_qr'
    )
    
    res["labels"] = cluster_model.fit_predict(similarity_matrix)
    res["k"] = k
    
    # 评估聚类质量
    if config.attributes.silhouette_score:
        # 使用原始特征计算轮廓系数
        res["silhouette"] = silhouette_score(feats, res["labels"])
        print(f"Silhouette Score: {res['silhouette']:.3f}")

    # 保留原有结构
    res["true_labels"] = true_labels
    res["ids"] = ids

    # 恢复模型状态
    for m in model.values():
        m.train()

    return res

def k_means(model, loader, config, device):
    # 创建一个字典来存储聚类结果
    res = {}

    # 将模型切换到评估模式
    for m in model:
        model[m].eval()

    # 初始化标签管理器，用于处理标签的转换
    labels_manager = LabelsManager(config)
    # 定义存储特征、ID和真实标签的列表
    feats = []
    ids = []
    true_labels = []
    
    # 确定K-means算法的最佳K值
    with torch.no_grad():
        print("K-means: loading training set")
        
        # 判断数据集是否为EK（视频数据集）
        if "ek" in config.data.dataset:
            # 定义用于存储起始和结束帧的列表
            start_frames = []
            stop_frames = []
            
            # 遍历数据加载器中的批次数据
            for batch_idx, (
                video_target,  # 输入视频数据
                y_target,  # 真实标签
                path,  # 文件路径
                start_frame,  # 视频的起始帧
                stop_frame,  # 视频的结束帧
                video_id,  # 视频的ID
            ) in enumerate(tqdm(loader)):  # 使用tqdm显示进度条

                # 对输入的视频数据进行重塑，使其符合模型的输入要求
                video_target = video_target.view(
                    (-1, config.data.num_segments, 3) + video_target.size()[-2:]
                )
                b, t, c, h, w = video_target.size()  # 获取批次大小、时间段、通道数、高度和宽度

                # 将视频数据移到GPU
                video_target = video_target.to(device).view(-1, c, h, w)

                # 获取视频嵌入（通过视频模型生成特征表示）
                video_embedding = model["video_model"](video_target)
                video_embedding = video_embedding.view(b, t, -1)  # 重塑嵌入的形状
                video_embedding = model["fusion_model"](video_embedding)  # 融合模型处理嵌入

                # 将每个视频的特征表示加入特征列表
                for i in range(b):
                    feats.append(
                        video_embedding[i]
                        .reshape(video_embedding.size(-1))  # 重塑为一维向量
                        .cpu()  # 移回CPU
                        .numpy()  # 转换为NumPy数组
                    )
                    # 记录每个视频的起始帧和结束帧
                    start_frames.append(start_frame[i].item())
                    stop_frames.append(stop_frame[i].item())
                    # 记录视频ID
                    ids.append(video_id[i].item())
                    # 转换标签并记录真实标签
                    true_labels.append(
                        labels_manager.convert_single_label(y_target[i].item())
                    )
            # 存储起始帧和结束帧信息
            res["start_frames"] = start_frames
            res["stop_frames"] = stop_frames
        else:
            # 如果数据集不是EK，按常规方式进行处理
            # for batch_idx, (video_target, y_target, path, video_id,skeleton) in enumerate(
            for batch_idx, (video_target, y_target, video_id,skeleton) in enumerate(
                tqdm(loader)  # 使用tqdm显示进度条
            ):
                # 对输入的视频数据进行重塑，使其符合模型的输入要求
                # print(video_target)
                video_target = video_target.view(
                    (-1, config.data.num_segments, 3) + video_target.size()[-2:]
                )
                b, t, c, h, w = video_target.size()  # 获取批次大小、时间段、通道数、高度和宽度

                # 将视频数据移到GPU
                video_target = video_target.to(device).view(-1, c, h, w)

                # 获取视频嵌入（通过视频模型生成特征表示）
                video_embedding = model["video_model"](video_target)
                video_embedding = video_embedding.view(b, t, -1)  # 重塑嵌入的形状
                video_embedding = model["fusion_model"](video_embedding)  # 融合模型处理嵌入

                # 将每个视频的特征表示加入特征列表
                for i in range(b):
                    feats.append(
                        video_embedding[i]
                        .reshape(video_embedding.size(-1))  # 重塑为一维向量
                        .cpu()  # 移回CPU
                        .numpy()  # 转换为NumPy数组
                    )
                    # 记录视频ID
                    ids.append(video_id[i].item())
                    # 转换标签并记录真实标签
                    true_labels.append(
                        labels_manager.convert_single_label(y_target[i].item())
                    )

    # 输出提示，开始进行聚类
    print("Performing clustering...")

    # 如果配置中要求使用肘部法则（Elbow Method）来确定最佳K值
    if config.attributes.use_elbow:
        print("Using elbow method to find optimal K")
        # 使用肘部法则来确定K值
        km = KMeans(algorithm='auto')
        visualizer = KElbowVisualizer(km, k=(4, 50))  # 设置K值的范围为4到50
        visualizer.fit(np.array(feats))  # 使用聚类特征进行拟合
        k = visualizer.elbow_value_  # 获取肘部法则确定的最佳K值
        print("K = {}".format(k))  # 打印最佳K值
    else:
        # 如果不使用肘部法则，则使用配置文件中指定的K值
        k = config.attributes.k_clustering

    # 使用K-means算法进行聚类
    res["labels"] = KMeans(
        n_clusters=k, random_state=0  # 设置聚类的数量为k，随机种子为0
    ).fit_predict(feats)  # 执行聚类并预测每个样本的标签

    # 存储真实标签
    res["true_labels"] = true_labels
    # 输出聚类完成的提示
    print("Clustering completed")
    # 存储视频ID
    res["ids"] = ids
    # 存储聚类的K值
    res["k"] = k

    # 将模型切换回训练模式
    for m in model:
        model[m].train()

    # 返回聚类结果
    return res

def extract(res, caption):
    # 将caption按空格分割成单词列表
    c = caption.split(" ")
    
    # 查找caption中所有"[MASK]"标记的索引位置
    indices = np.where(np.array(c) == "[MASK]")[0]
    
    # 返回res中对应位置的单词，使用"_"连接
    return "_".join([res.split()[index] for index in indices])

def tf_idf(most_commons):
    # 创建TfidfVectorizer对象，用于计算TF-IDF
    vectorizer = TfidfVectorizer()
    
    # 对最常见的词语列表进行TF-IDF计算，返回词向量
    vectors = vectorizer.fit_transform(most_commons)
    
    # 获取TF-IDF计算后的特征名（即词语）
    feature_names = vectorizer.get_feature_names_out()  # 使用 get_feature_names_out() 替代 get_feature_names()
    
    # 将TF-IDF矩阵转换为稠密矩阵
    dense = vectors.todense()
    
    # 将稠密矩阵转换为列表形式
    denselist = dense.tolist()
    
    # 将列表转换为DataFrame，并转置，使得每一列为一个特征词的TF-IDF值
    df = pd.DataFrame(denselist, columns=feature_names).T
    
    return df

def get_most_commons(input_dict, config):
    # 初始化一个空列表，用于存储每个标签的最常见词
    most_commons = []
    
    # 对输入字典的标签按数字大小进行排序，逐一处理
    for label in sorted(input_dict, key=lambda x: int(x)):
        # 计算每个标签下词语的出现频率
        counter = Counter(input_dict[label])
        
        # 选择出现频率最高的n_attributes个词并拼接成一个字符串
        most_common = " ".join(
            [word for word, _ in counter.most_common(config.attributes.n_attributes)]
        )
        
        # 将最常见的词字符串添加到列表中
        most_commons.append(most_common)
    
    return most_commons

# 提取每个聚类的属性及其对应的真实标签统计
def extract_cluster_attributes(clustering_results, config, epoch, run_id, elbow=None):
    # 定义属性文件路径，该文件包含目标域的数据集相关的属性信息
    attribute_file_path = "/opt/data/private/3-30第二次hodge聚类编写/utils/attributes/{}_target_{}.json".format(
        config.data.target_dataset, config.attributes.n_blanks
    )

    # 打开并加载属性文件
    with open(attribute_file_path, "r") as attribute_file:
        attributes = load(attribute_file)
    
    # 初始化用于存储聚类结果的字典
    attributes_per_cluster_label = defaultdict(list)#记录每个聚类中的图片的真实属性标签
    true_labels_per_cluster = defaultdict(list)#记录每个聚类中的图片的预测属性标签

    # 遍历聚类结果，根据每个样本的聚类标签将属性归类
    for i in range(len(clustering_results["labels"])):
        cluster_label = clustering_results["labels"][i]  # 获取当前样本的聚类标签
        # 将每个样本的属性添加到对应聚类标签的属性列表中
        for attr in attributes[str(clustering_results["ids"][i])]:
            # 将预先获得的每一张图片的属性添加到对应的聚类中
            attributes_per_cluster_label[cluster_label].append(attr)
            # 同时记录每个聚类中的图片获得的预测属性标签
            true_labels_per_cluster[str(cluster_label)].append(
                clustering_results["true_labels"][i]
            )

    # 获取每个聚类的最常见属性
    most_commons = get_most_commons(attributes_per_cluster_label, config)
    
    # 使用TF-IDF计算属性的重要性，并转换为DataFrame格式
    df = tf_idf(most_commons)
    
    # 存储每个聚类对应的最终属性
    attributes_per_cluster = {}
    
    # 遍历每个聚类，选择相应的属性
    for i in range(clustering_results["k"]):
        if config.attributes.selection == "topk":
            # 使用topk方法选择前k个最重要的属性
            attributes = (
                df[i].nlargest(config.attributes.tf_idf_topk_target).index.tolist()
            )
        elif config.attributes.selection == "threshold":
            # 使用阈值方法选择TF-IDF值大于某个阈值的属性
            attributes = df[df[i] > config.attributes.tf_idf_threshold].index.tolist()
        else:
            # 如果选择策略不认识，则抛出异常
            raise ValueError(
                "Attribute selection strategy {} not recognized!".format(
                    config.attributes.selection
                )
            )
        
        # 将选定的属性添加到每个聚类的属性列表中
        attributes_per_cluster[i] = attributes

        # 将当前epoch的聚类属性写入文件，便于后续分析
        with open("{}_attributes.txt".format(run_id), "a") as attributes_file:
            attributes_file.write(
                "\n\n{}: original: {}\n, middle: {}\n, final: {}\n".format(
                    epoch,
                    attributes_per_cluster_label,  # 原始聚类标签属性
                    most_commons,  # 中间步骤的最常见属性
                    attributes_per_cluster,  # 最终选出的属性
                )
            )

    # 生成每个聚类的真实标签统计信息
    generic_true_labels_per_cluster = defaultdict(dict)
    for cluster_label in true_labels_per_cluster:
        inner_dict = {}
        # 获取该聚类标签下所有真实标签的集合
        labels_set = list(set(true_labels_per_cluster[cluster_label]))
        # 统计每个真实标签出现的次数
        for label in labels_set:
            inner_dict[label] = true_labels_per_cluster[cluster_label].count(label)
        # 将统计结果存入字典
        generic_true_labels_per_cluster[cluster_label] = inner_dict

    # 返回最终每个聚类的属性及其对应的真实标签统计
    return attributes_per_cluster, generic_true_labels_per_cluster

# 从指定的数据集中提取每个类别对应的属性，并根据配置对这些属性进行筛选。
def extract_class_attributes(config):
    # 创建一个标签管理器，用于将标签转换为适合的格式
    labels_manager = LabelsManager(config)

    # 根据配置中的源数据集名称和属性空白数量，构造属性文件路径
    # 该文件包含了源数据集的每个样本的属性信息
    attribute_file_path = "/opt/data/private/3-30第二次hodge聚类编写/utils/attributes/{}_source_{}.json".format(
        config.data.source_dataset, config.attributes.n_blanks
    )

    # 打开并加载属性文件
    with open(attribute_file_path, "r") as attribute_file:
        attributes = load(attribute_file)

    # 创建一个字典，用于存储每个类别对应的属性列表
    attributes_per_class = defaultdict(list)

    # 打开源数据集的训练集文件，逐行读取每个样本
    with open(config.data.source_train_file, "r") as txt_file:
        # 判断是否为EPIC-Kitchens数据集，如果是，则需要特别处理
        epic_kitchens = "ek" in config.data.dataset

        # 遍历训练集中的每一行数据
        for i, line in enumerate(txt_file):
            if epic_kitchens:
                # 对于EPIC-Kitchens数据集，每行数据的格式为: 视频路径 空格 标签
                _, _, _, label = line.split()
            else:
                # 对于其他数据集，每行数据的格式为: 视频路径 空格 标签
                _, label = line.split()

            # 根据当前行的样本索引，将该视频对应的属性加入到该标签的属性列表中
            for attr in attributes[str(i)]:
                attributes_per_class[label].append(attr)

        # 获取每个类别最常见的属性
        most_commons = get_most_commons(attributes_per_class, config)

        # 使用TF-IDF方法计算每个属性的重要性
        df = tf_idf(most_commons)

        # 创建一个新的字典，用于存储筛选后的每个类别的属性
        clean_attributes_per_class = {}

        # 对每个类别的属性进行筛选，根据配置选择最重要的属性
        for i in range(len(attributes_per_class.keys())):
            if config.attributes.selection == "topk":
                # 如果选择策略为"topk"，则选择TF-IDF值前K个最重要的属性
                attributes = (
                    df[i].nlargest(config.attributes.tf_idf_topk_source).index.tolist()
                )
            elif config.attributes.selection == "threshold":
                # 如果选择策略为"threshold"，则选择TF-IDF值大于某个阈值的属性
                attributes = df[
                    df[i] > config.attributes.tf_idf_threshold
                ].index.tolist()
            else:
                # 如果选择策略不识别，则抛出异常
                raise ValueError(
                    "Attribute selection strategy {} not recognized!".format(
                        config.attributes.selection
                    )
                )
            # 将筛选后的属性存储到字典中，并转换标签为适合的格式
            clean_attributes_per_class[
                labels_manager.convert_single_label(i)
            ] = attributes

    # 返回清洗后的每个类别对应的属性
    return clean_attributes_per_class

def compute_score(s, t, weights):
    # 如果源域属性与目标簇属性相同，返回1，否则根据位置差异计算加权得分
    if s == t:
        return 1
    else:
        return 1 * weights[abs(t - s)]  # 通过权重来计算不相同属性的得分，权重由位置差计算得出

# 该函数的作用是执行源域属性与目标域簇的匹配，根据一定的匹配阈值返回匹配结果。
# 主要功能是在源域和目标域的特征之间进行对比，计算匹配度，并将每个目标簇（cluster）与一个源标签进行匹配。
def match_attributes(
    source_attributes_per_class, target_attributes_per_cluster, config
):
    # 存储每个目标簇标签的匹配结果
    matches_per_cluster_label = {}

    # 遍历每个目标簇
    for cluster_label in target_attributes_per_cluster:
        target_attributes = target_attributes_per_cluster[cluster_label]  # 获取目标簇的属性列表

        # 存储每个目标簇与源标签的匹配结果
        matched_source_labels = []
        
        # 遍历源域每个标签
        for source_label in source_attributes_per_class:
            source_attributes = source_attributes_per_class[source_label]  # 获取源标签的属性列表

            # 对源标签的属性进行反转，并计算权重
            ref = np.flip(np.array(list(range(len(source_attributes)))))  # 反转源属性的索引
            weights = (ref - np.min(ref)) / (np.max(ref) - np.min(ref))  # 计算权重（归一化）

            score = 0  # 初始化得分

            # 对源标签的每个属性与目标簇的每个属性进行匹配
            for s in range(len(source_attributes)):
                for t in range(len(target_attributes)):
                    if target_attributes[t] == source_attributes[s]:  # 如果属性匹配
                        score += weights[abs(t - s)]  # 根据位置差异加权得分

            # 计算最终得分并归一化
            score /= len(source_attributes)

            # 判断得分是否超过匹配阈值，如果超过则认为匹配成功
            match = score > config.attributes.matching_threshold
            if match:# 如果开启了日志记录，输出匹配信息
                if config.logging.verbose:
                    print("------------------------------------------")
                    print("MATCH: score = {}".format(score))
                    print("SA({}) = {}".format(source_label, source_attributes))  # 源标签和属性
                    print("TA = {}".format(target_attributes))  # 目标簇的属性
                    print("------------------------------------------")
                matched_source_labels.append((source_label, score))  # 将匹配的源标签和得分保存

        # 如果有匹配的源标签
        if len(matched_source_labels):
            # 找到得分最高的源标签
            max_confidence = max([conf for _, conf in matched_source_labels])
            # 找出得分最高的所有源标签
            top_confident_labels = [
                label for label, conf in matched_source_labels if conf == max_confidence
            ]
            # 如果有多个得分最高的标签，则随机选择一个
            if len(top_confident_labels) > 1:
                candidate_label = choice(top_confident_labels)
            else:
                candidate_label = top_confident_labels[0]  # 否则选择唯一的标签
            matches_per_cluster_label[cluster_label] = candidate_label  # 将匹配的源标签分配给目标簇

    # 记录开放集标签、已匹配簇标签和未匹配簇标签
    open_set_labels = []  # 存储开放集标签
    matched_cluster_labels = []  # 存储已匹配的目标簇标签
    unmatched_cluster_labels = []  # 存储未匹配的目标簇标签

    # 遍历目标簇标签，判断是否有匹配结果
    for cluster_label in target_attributes_per_cluster:
        if cluster_label not in matches_per_cluster_label:
            # 如果当前目标簇没有匹配结果，则将其标记为开放集标签
            open_set_label = " and ".join(
                target_attributes_per_cluster[cluster_label][
                    : config.attributes.final_prompt_length
                ]
            )
            matches_per_cluster_label[cluster_label] = open_set_label  # 给目标簇分配开放集标签
            if open_set_label not in open_set_labels:
                open_set_labels.append(open_set_label)  # 将开放集标签添加到开放集标签列表
                unmatched_cluster_labels.append(cluster_label)  # 将目标簇添加到未匹配簇列表
        else:
            matched_cluster_labels.append(cluster_label)  # 如果有匹配，则将其添加到已匹配簇列表

    # 返回匹配结果
    res = {
        "matches_per_cluster_label": matches_per_cluster_label,  # 每个目标簇的匹配标签
        "open_set_labels": open_set_labels,  # 开放集标签
        "matched_cluster_labels": matched_cluster_labels,  # 已匹配的簇标签
        "unmatched_cluster_labels": unmatched_cluster_labels,  # 未匹配的簇标签
    }

    return res  # 返回匹配结果

def compute_clustering_accuracy(
    matches_per_cluster_label, clustering_labels, true_labels, config
):
    # 创建一个标签管理器，用于标签的转换（比如将标签从数字映射到字符串）
    labels_manager = LabelsManager(config)

    pred = []  # 用于存储预测的标签列表

    # 遍历每个聚类结果中的标签
    for i in range(len(clustering_labels)):
        assigned_label = clustering_labels[i]  # 获取当前样本的聚类标签
        found = False  # 标记是否找到匹配的源标签
        
        # 遍历目标簇标签及其对应的匹配源标签
        for cluster_label in matches_per_cluster_label:
            if assigned_label == cluster_label:  # 如果当前聚类标签与目标簇标签匹配
                # 判断是否有与当前聚类标签对应的源标签
                if assigned_label in labels_manager.rev_label_map:
                    # 如果有源标签，则将其转换为相应的标签并添加到预测结果列表中
                    pred.append(
                        labels_manager.convert_single_label(
                            assigned_label, reverse=True
                        )
                    )
                else:
                    # 如果没有找到对应的源标签，则将预测标签标记为未知标签（"UNK"）
                    pred.append(
                        labels_manager.convert_single_label("UNK", reverse=True)
                    )
                found = True  # 标记已经找到匹配标签
        # 如果没有找到匹配的标签，抛出异常
        assert found

    # 确保预测标签的数量与聚类标签的数量一致
    assert len(pred) == len(clustering_labels)

    # 将真实标签转换为相应的标签
    true = [labels_manager.convert_single_label(l, reverse=True) for l in true_labels]

    # 计算预测结果与真实标签的准确率
    acc = accuracy_score(pred, true)

    return acc  # 返回准确率
