# HiPose-CLIP: CLIP-Based Hierarchical Multimodal Alignment for Open-Set Video Domain Adaptation

本仓库是论文 **"HiPose-CLIP: 基于 CLIP 的分层多模态对齐开放集视频域自适应"** 的官方代码实现（北京理工大学毕业设计项目） 。

## 📖 概述

针对开放集视频无监督领域自适应 (OUVDA) 任务中存在的**模态间语义偏差**与**领域异构性**问题，本项目提出了 **HiPose-CLIP** 框架 。

该框架通过融合视觉语义与人体骨骼时序特征，构建了一个双流时空增强架构，能够有效应对光照变化、视角偏移等复杂场景下的跨域迁移挑战 。

## ✨ 核心贡献

1. 
**分层多模态融合架构**：利用冻结的 CLIP 视觉编码器保留原始语义，配合可学习的骨骼时序分支捕捉运动模式 。


2. 
**动态分层掩码跨模态注意力 (DHM-CMA)**：采用时序弹性对齐与多头门控机制，解决视频与骨骼序列的帧率差异，实现时空自适应融合 。


3. 
**基于 Hodge 分解的拓扑聚类**：利用势场分析与旋度流检测识别目标域中的未知类别，显著提升了开放集检测的鲁棒性 。


4. 
**动态提示学习 (Dynamic Prompting)**：结合 VQA 视觉定位生成“阶段化骨骼运动描述”，将视觉语义精准注入文本空间 。



## 🛠️ 方法架构

HiPose-CLIP 的工作流程分为三个核心阶段 ：

* 
**特征建模**：利用预训练 GCN 提取 17 个关键点骨骼序列，并通过运动能量准则进行主体筛选 。


* 
**层次融合**：DHM-CMA 模块通过关键帧选择或线性插值实现时序对齐，并使用门控注意力抑制噪声模态的干扰 。


* 
**开放集发现**：改进的 Hodge-Helmholtz 分解聚类算法，将时间复杂度从 $O(N^3)$ 降至 $O(Nk^2)$，高效识别簇间边界与异常模式 。



## 📊 实验结果

在多个跨域基准数据集上，HiPose-CLIP 均达到了 SOTA 性能 ：

| 任务 | 指标 (H-Score) | 较基线提升 (vs AutoLabel) | 未知类检测准确率 |
| --- | --- | --- | --- |
| **HMDB → UCF** | **98.9%** | +9.8% | 100% |
| **UCF → Olympic** | **93.1%** | +7.6% | 100% |

> 
> **消融实验证实**：分层融合结构贡献了 12.7% 的性能增益，Hodge 聚类贡献了 8.3% 。
> 
> 

## 🚀 快速开始

*(注：以下为标准项目结构建议)*

### 环境要求

* Python 3.8+
* PyTorch 2.0+
* CUDA 11.7+
* `pyskl` (用于骨骼提取)

### 安装

```bash
git clone https://github.com/YourUsername/HiPose-CLIP.git
cd HiPose-CLIP
pip install -r requirements.txt

```

### 数据准备

请参考 `data/README.md` 下载并预处理以下数据集：

* HMDB51 & UCF101 


* Olympic Sports 



### 运行训练

```bash
python train.py --config configs/hmdb_ucf.yaml

```

## 📜 引用

如果您觉得本研究对您有帮助，请引用我们的工作：

```bibtex
@article{fu2025hipose,
  title={HiPose-CLIP: 基于 CLIP 的分层多模态对齐开放集视频域自适应},
  author={傅裕},
  school={北京理工大学},
  year={2025}
}

```

---

*本项目由北京理工大学计算机学院人工智能专业傅裕完成，指导老师：逢金辉（北京理工大学）、许悦聪（新加坡国立大学）。*
