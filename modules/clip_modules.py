import torch
from torch import nn


class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, prompts, tokenized_prompts=None, learnable_prompts=False):
        if learnable_prompts:
            orig_shape = prompts.shape  # 保存原始形状 [B, Cls, Token, Dim]
            # 合并Batch和Cls维度
            prompts = prompts.view(-1, *orig_shape[2:])  # => [B*Cls, Token, Dim]
            # 扩展tokenized prompts
            tokenized = tokenized_prompts.repeat(orig_shape[0], 1)  # [B*Cls, Token]

            text_features = self.model.encode_text_with_prompts(prompts, tokenized)
            return text_features.view(orig_shape[0], orig_shape[1], -1)
        return self.model.encode_text(prompts)


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 确保输入数据是float32类型
        dtype = next(self.model.parameters()).dtype  # 检查模型权重的 dtype
        image = image.to(device)  # 将输入调整为模型权重的 dtype
        # print(f"Input dtype: {image.dtype}, model dtype: {dtype}")  # Debug 信息
        return self.model.encode_image(image)



