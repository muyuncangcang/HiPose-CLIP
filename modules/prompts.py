import math
import torch
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch import nn
from collections import OrderedDict
from csv import reader
from collections import defaultdict
from random import choice, shuffle
from ast import literal_eval
import torch.nn.functional as F

_tokenizer = _Tokenizer()


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

def trunc_normal_(x, mean=0.0, std=1.0):
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)

class TAggregate(nn.Module):
    def __init__(self, clip_length=None, embed_dim=2048, n_layers=6):
        super(TAggregate, self).__init__()
        self.clip_length = clip_length
        drop_rate = 0.0
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)
        self.transformer_enc = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, norm=nn.LayerNorm(embed_dim)
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, clip_length + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        with torch.no_grad():
            trunc_normal_(self.pos_embed, std=0.02)
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        nvids = x.shape[0]

        cls_tokens = self.cls_token.expand(nvids, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x.transpose_(1, 0)
        o = self.transformer_enc(x)

        return o[0]

class TemporalTransformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks((x))

class visual_prompt(nn.Module):
    def __init__(self, sim_head, clip_state_dict, T):
        super().__init__()
        self.sim_header = sim_head
        self.T = T
        assert sim_head in ["meanP", "LSTM", "Transf", "Conv_1D", "Transf_cls"]

        if (
            self.sim_header == "LSTM"
            or self.sim_header == "Transf"
            or self.sim_header == "Transf_cls"
            or self.sim_header == "Conv_1D"
        ):
            embed_dim = clip_state_dict["text_projection"].shape[1]

            context_length = clip_state_dict["positional_embedding"].shape[0]
            vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
            transformer_width = clip_state_dict["ln_final.weight"].shape[0]
            transformer_heads = transformer_width // 64

            transformer_layers = len(
                set(
                    k.split(".")[2]
                    for k in clip_state_dict
                    if k.startswith(f"transformer.resblocks")
                )
            )

            self.frame_position_embeddings = nn.Embedding(context_length, embed_dim)
        if self.sim_header == "Transf":
            self.transformer = TemporalTransformer(
                width=embed_dim, layers=6, heads=transformer_heads
            )
            # print("layer=6")
        if self.sim_header == "LSTM":
            self.lstm_visual = nn.LSTM(
                input_size=embed_dim,
                hidden_size=embed_dim,
                batch_first=True,
                bidirectional=False,
                num_layers=1,
            )

        self.apply(self.init_weights)

        if self.sim_header == "Transf_cls":
            self.transformer = TAggregate(
                clip_length=self.T, embed_dim=embed_dim, n_layers=6
            )

        if self.sim_header == "Conv_1D":
            self.shift = nn.Conv1d(
                embed_dim, embed_dim, 3, padding=1, groups=embed_dim, bias=False
            )
            weight = torch.zeros(embed_dim, 1, 3)
            weight[: embed_dim // 4, 0, 0] = 1.0
            weight[embed_dim // 4 : embed_dim // 4 + embed_dim // 2, 0, 1] = 1.0
            weight[-embed_dim // 4 :, 0, 2] = 1.0
            self.shift.weight = nn.Parameter(weight)

    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if "beta" in dir(module) and "gamma" in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        b, t, c = x.size()
        x = x.contiguous()
        if self.sim_header == "meanP":
            pass
        elif self.sim_header == "Conv_1D":
            x_original = x
            x = x.view(-1, c, t)
            x = self.shift(x.float())
            x = x.permute(0, 2, 1)
            x = x.type(x_original.dtype) + x_original

        elif self.sim_header == "Transf":
            x_original = x
            seq_length = t
            position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            x = x + frame_position_embeddings

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = x.type(x_original.dtype) + x_original

        elif self.sim_header == "LSTM":
            x_original = x
            x, _ = self.lstm_visual(x.float())
            self.lstm_visual.flatten_parameters()
            x = torch.cat((x, x_original[:, x.size(1) :, ...].contiguous()), dim=1)
            x = x.type(x_original.dtype) + x_original
        elif self.sim_header == "Transf_cls":
            x_original = x
            return self.transformer(x).type(x_original.dtype)

        else:
            raise ValueError("Unknown optimizer: {}".format(self.sim_header))
        return x.mean(dim=1, keepdim=False)

class EKClassAug:
    def __init__(self, config):
        self.class_augs = defaultdict(list)

        if config.data.class_augmenter.version == "training_set":
            csv_file_path = "data/EPIC_100_train.csv"
            with open(csv_file_path, "r") as csv_file:
                csv_reader = reader(csv_file)
                line_count = 0
                for row in csv_reader:

                    if line_count == 0:
                        fields = row
                        line_count += 1

                    else:
                        verb = row[fields.index("verb")]
                        narration = row[fields.index("narration")]
                        if narration not in self.class_augs[verb]:
                            self.class_augs[verb].append(narration)
        else:
            csv_file_path = "data/EPIC_100_verb_classes.csv"
            with open(csv_file_path, "r") as csv_file:
                csv_reader = reader(csv_file)
                line_count = 0
                for row in csv_reader:

                    if line_count == 0:
                        fields = row
                        line_count += 1

                    else:
                        verb = row[fields.index("key")]
                        instances = row[fields.index("instances")]
                        instances = literal_eval(instances)
                        self.class_augs[verb] = instances

        self.label_map = {
            "opening": "open",
            "taking": "take",
            "closing": "close",
            "putting": "put",
            "washing": "wash",
            "pouring": "pour",
            "mixing": "mix",
            "cutting": "cut",
        }

        self.min_len = min(
            [len(self.class_augs[verb]) for _, verb in self.label_map.items()]
        )

    def produce_augmentation(self, verb):
        root_verb = self.label_map[verb]
        options = self.class_augs[root_verb]
        res = choice(options)
        return res

    def aug_list(self, verb):
        root_verb = self.label_map[verb]
        options = self.class_augs[root_verb]
        shuffle(options)
        return options[: self.min_len]

def apply_cutmix(text_aug, config, classes):

    res = []
    for t in text_aug:
        class_instances = []
        for _, c in classes:
            instance = "{} {} {}".format(t, config.data.cutmix.connector, c)
            class_instances.append(instance)
        res.append(class_instances)

    return res

def plug_domain_positives(text_aug, config, eval=False, target=False):
    assert config.loss.target.prompts.prompt_improvements <= 2
    res = []
    ek2_improvements = ["on a wooden background"]
    ek1_improvements = ["on a black background"]
    ek3_improvements = ["on a tiled background"]

    if not target:
        if config.data.cutmix.enabled and not eval:
            for txts in text_aug:
                texts = []
                for txt in txts:
                    if "kinetics-nec" in config.data.dataset:
                        texts.append("{} in a gym".format(txt))
                    elif config.data.dataset.startswith(
                        "ek"
                    ) and config.data.dataset not in [
                        "ek24",
                        "ek42",
                    ]:

                        if config.data.dataset.startswith("ek2"):
                            texts.append("{} {}".format(txt, choice(ek2_improvements)))
                        elif config.data.dataset.startswith("ek1"):
                            texts.append("{} {}".format(txt, choice(ek1_improvements)))
                        elif config.data.dataset.startswith("ek3"):
                            texts.append("{} {}".format(txt, choice(ek3_improvements)))
                        else:
                            print("EK dataset not recognized for domain positives!")
                            exit(1)
                res.append(texts)
        else:
            if "kinetics-nec" in config.data.dataset:
                res = ["{} in a gym".format(t) for t in text_aug]
            elif config.data.dataset.startswith("ek") and config.data.dataset not in [
                "ek24",
                "ek42",
            ]:
                if config.data.dataset.startswith("ek2"):
                    res = [
                        "{} {}".format(t, choice(ek2_improvements)) for t in text_aug
                    ]
                elif config.data.dataset.startswith("ek1"):
                    res = [
                        "{} {}".format(t, choice(ek1_improvements)) for t in text_aug
                    ]
                elif config.data.dataset.startswith("ek3"):
                    res = [
                        "{} {}".format(t, choice(ek3_improvements)) for t in text_aug
                    ]
                else:
                    print("EK dataset not recognized for domain positives!")
                    exit(1)
            else:
                print("Dataset not recognized for domain positives!")
                exit(1)
    else:
        if config.data.cutmix.enabled and not eval:
            for txts in text_aug:
                texts = []
                for txt in txts:
                    if "kinetics-nec" in config.data.dataset:
                        texts.append("{} in a gym".format(txt))
                    elif config.data.dataset.startswith(
                        "ek"
                    ) and config.data.dataset not in [
                        "ek24",
                        "ek42",
                    ]:

                        if config.data.dataset.endswith("2"):
                            texts.append("{} {}".format(txt, choice(ek2_improvements)))
                        elif config.data.dataset.endswith("1"):
                            texts.append("{} {}".format(txt, choice(ek1_improvements)))
                        elif config.data.dataset.endswith("3"):
                            texts.append("{} {}".format(txt, choice(ek3_improvements)))
                        else:
                            print("EK dataset not recognized for domain positives!")
                            exit(1)
                res.append(res)
        else:
            if "kinetics-nec" in config.data.dataset:
                res = ["{} in a gym".format(t) for t in text_aug]
            elif config.data.dataset.startswith("ek") and config.data.dataset not in [
                "ek24",
                "ek42",
            ]:
                if config.data.dataset.endswith("2"):
                    res = [
                        "{} {}".format(t, choice(ek2_improvements)) for t in text_aug
                    ]
                elif config.data.dataset.endswith("1"):
                    res = [
                        "{} {}".format(t, choice(ek1_improvements)) for t in text_aug
                    ]
                elif config.data.dataset.endswith("3"):
                    res = [
                        "{} {}".format(t, choice(ek3_improvements)) for t in text_aug
                    ]
                else:
                    print("EK dataset not recognized for domain positives!")
                    exit(1)
            else:
                print("Dataset not recognized for domain positives!")
                exit(1)
    return res

def text_prompt(classes):

    text_dict = {}

    text_aug = [
        f"a photo of action {{}}",
        f"a picture of action {{}}",
        f"Human action of {{}}",
        f"{{}}, an action",
        f"{{}} this is an action",
        f"{{}}, a video of action",
        f"Playing action of {{}}",
        f"{{}}",
        f"Playing a kind of action, {{}}",
        f"Doing a kind of action, {{}}",
        f"Look, the human is {{}}",
        f"Can you recognize the action of {{}}?",
        f"Video classification of {{}}",
        f"A video of {{}}",
        f"The man is {{}}",
        f"The woman is {{}}",
    ]

    num_text_aug = len(text_aug)
    for i, txt in enumerate(text_aug):
        text_dict[i] = torch.cat([clip.tokenize(txt.format(c)) for _, c in classes])

    text_dict_target = text_dict

    classes = torch.cat([v for _, v in text_dict.items()])

    res = {
        "classes": classes,
        "num_text_aug": num_text_aug,
        "text_dict": text_dict,
        "text_dict_target": text_dict_target,
        "text_aug": text_aug,
    }

    return res

def text_prompt_domain(dataset):

    if dataset == "kinetics-nec":

        positives = ["sky", "people at the sea", "people on the grass", "airplane"]

        negatives = [
            "people in a gym",
            "gym",
            "people moving in a gym",
            "action in a gym",
        ]

    elif "ek" in dataset:

        positives = ["sky", "people at the sea", "people on the grass", "airplane"]

        negatives = [
            "people in a kitchen",
            "kitchen",
            "hands in the kitchen",
            "sink and stove",
        ]

    else:
        positives = None
        negatives = None
        print("Target dataset not recognized!")
        exit(1)

    num_text_aug_positives = len(positives)
    num_text_aug_negatives = len(negatives)

    text_dict_positives = {}
    for i, txt in enumerate(positives):
        text_dict_positives[i] = clip.tokenize(txt)

    text_dict_negatives = {}
    for i, txt in enumerate(negatives):
        text_dict_negatives[i] = clip.tokenize(txt)

    text_dict = {"positives": text_dict_positives, "negatives": text_dict_negatives}
    num_text_aug = {
        "positives": num_text_aug_positives,
        "negatives": num_text_aug_negatives,
    }

    res = (num_text_aug, text_dict)

    return res



def construct_prompts(ctx, prefix, suffix, label=None):
    if label is not None:
        prefix = prefix[label]
        suffix = suffix[label]

    # print(f"ctx shape: {ctx.shape}")          # ctx shape: torch.Size([17, 4, 512])
    # print(f"prefix shape: {prefix.shape}")    # prefix shape: torch.Size([23, 1, 512])
    # print(f"suffix shape: {suffix.shape}")    # suffix shape: torch.Size([23, 72, 512])

    prompts = torch.cat(
        [
            prefix,  # (dim0, 1, dim)
            ctx,  # (dim0, n_ctx, dim)
            suffix,  # (dim0, *, dim)
        ],
        dim=1,
    )

    return prompts

class MetaNet(nn.Module):
    def __init__(self, vis_dim, ctx_dim):
        super().__init__()
        self.fc1 = nn.Linear(vis_dim, vis_dim // 16)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(vis_dim // 16, ctx_dim)

    def forward(self, x):
        x = x.float()
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class CoCoOpPromptLearner(nn.Module):
    # dim0 is either batch_size (during training) or n_cls (during testing)
    # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
    # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
    # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)
    def __init__(self, config, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = config.learnable_prompts.n_context
        self.ctx_init = config.learnable_prompts.context_init
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        self.token_embedding = clip_model.token_embedding
        self.clip_model = clip_model

        if self.ctx_init != "none":
            # use given words to initialize context vectors
            self.ctx_init = self.ctx_init.replace("_", " ")
            n_ctx = len(self.ctx_init.split(" "))
            prompt = clip.tokenize(self.ctx_init)
            prompt = prompt.cuda()
            clip_model = clip_model.cuda()
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = self.ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = MetaNet(vis_dim, ctx_dim)
        classnames = [name for idx, name in classnames]  # 先提取纯字符串
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat(
            [clip.tokenize(p) for p in prompts]
        )  # (n_cls, n_tkn)
        tokenized_prompts = tokenized_prompts.cuda()
        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts).type(self.dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def _build_prompts(self, open_set_labels):
        # 处理新类别名称
        new_classnames = [name.replace("_", " ") for name in open_set_labels]
        if self.ctx_init != "none":
            prompt_prefix = self.ctx_init
            n_ctx = self.n_ctx  # 使用类初始化时设置的n_ctx
        else:
            prompt_prefix = " ".join(["X"] * self.n_ctx)  # 保持相同数量的上下文占位符
        new_prompts = [f"{prompt_prefix} {name}." for name in open_set_labels]

        # 生成新的tokenized prompts
        new_tokenized_prompts = torch.cat([clip.tokenize(p) for p in new_prompts]).to(self.tokenized_prompts.device)
        
        # 合并到原有tokenized_prompts
        # self.tokenized_prompts = torch.cat([self.tokenized_prompts, new_tokenized_prompts], dim=0)
        self.tokenized_prompts = new_tokenized_prompts
        
        # 计算新类别的嵌入
        with torch.no_grad():
            new_embedding = self.token_embedding(new_tokenized_prompts).type(self.dtype)
        
        # 合并token_prefix和token_suffix
        new_token_prefix = new_embedding[:, :1, :]
        new_token_suffix = new_embedding[:, 1 + self.n_ctx :, :]
        # self.token_prefix = torch.cat([self.token_prefix, new_token_prefix], dim=0)
        # self.token_suffix = torch.cat([self.token_suffix, new_token_suffix], dim=0)
        self.token_prefix = new_token_prefix
        self.token_suffix = new_token_suffix

        # 更新类别数量
        self.n_cls = len(new_classnames)
        # 更新名称长度（如果需要）
        new_name_lens = [len(_tokenizer.encode(name)) for name in new_classnames]
        self.name_lens.extend(new_name_lens)

    def forward(self, im_features):
        im_features = im_features.mean(dim=1)  # 全局平均池化
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx  # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        # print(f"im_features shape: {im_features.shape}")
        # print(f"ctx_shifted shape: {ctx_shifted.shape}")
        # print(f"prefix shape: {prefix.shape}")
        # print(f"suffix shape: {suffix.shape}")

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts #torch.Size([6, 6, 77, 512])[batch_size, num_classes, num_tokens, feature_dim]



class CrossModalFusion(nn.Module):
    def __init__(self, clip_dim=512, pose_dim=64, num_heads=4):
        super().__init__()
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, clip_dim),
            nn.GELU(),
            nn.LayerNorm(clip_dim)
        )
        self.attn = nn.MultiheadAttention(embed_dim=clip_dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(clip_dim)

    def forward(self, clip_feat, pose_feat):
        clip_feat = clip_feat.float()
        pose_feat = pose_feat.float()
        # --- 投影骨骼特征到与CLIP特征同一空间 ---
        # print(f"[DEBUG] clip_feat shape: {clip_feat.shape}")   # 应为 [B, T, 512]torch.Size([6, 8, 512])
        # print(f"[DEBUG] pose_feat shape: {pose_feat.shape}")   # 检查骨架特征原始形状torch.Size([6, 165, 64])
        pose_proj = self.pose_proj(pose_feat)
        # print(f"[DEBUG] pose_proj shape: {pose_proj.shape}")   # 应为 [B, T, 512]torch.Size([6, 165, 512])
        
        # --- 跨模态注意力 (CLIP作为Query，骨骼作为Key/Value) ---
        # print(f"[DEBUG] query shape: {clip_feat.permute(1,0,2).shape}")  # 预期 [T, B, 512]torch.Size([8, 6, 512])
        fused, _ = self.attn(
            query=clip_feat.permute(1,0,2),  # [T,B,D]
            key=pose_proj.permute(1,0,2),
            value=pose_proj.permute(1,0,2)
        )  # [T,B,D]
        
        fused = fused.permute(1,0,2)  # [B,T,D]
        
        # --- 残差连接与层归一化 (保持与visual_prompt相似的结构) ---
        fused = self.norm(clip_feat + fused)  # [B,T,D]
        
        return fused

# class PoseEncoder(nn.Module):# [B, T, hidden]
#     def __init__(self, input_dim=2, hidden=64):
#         super().__init__()
#         self.joint_emb = nn.Sequential(
#             nn.Linear(17*input_dim, 256),
#             nn.GELU(),
#             nn.LayerNorm(256),
#             nn.Linear(256, hidden)
#         )
#         self.temporal_conv = nn.Conv1d(hidden, hidden, 3, padding=1)
        
#     def forward(self, x):
#         # x: [B, T, V, C]
#         B, T, V, C = x.size()
#         x = x.view(B, T, -1)        # [B, T, 34]
#         return self.joint_emb(x)  # [B, T, hidden]

class PoseEncoderGCN(nn.Module):
    def __init__(self, input_dim=2, hidden=64, num_joints=17):
        super().__init__()
        self.num_joints = num_joints
        self.joint_emb = nn.Sequential(
            GCNBlock(input_dim, 64, adj=self._build_adjacency()),  # 图卷积层
            GCNBlock(64, hidden, adj=self._build_adjacency()),
            nn.Dropout(0.1)
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden*num_joints, hidden, 3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True)
        )
        
    def _build_adjacency(self):
        # 人体骨骼连接定义（根据COCO-17关键点）
        bones = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 躯干
            (5, 7), (7, 9), (6, 8), (8, 10),  # 手臂
            (11, 13), (13, 15), (12, 14), (14, 16)  # 腿
        ]
        adj = torch.eye(self.num_joints)
        for i, j in bones:
            adj[i, j] = 1
            adj[j, i] = 1  # 无向图
        return self._normalize_adj(adj)
    
    def _normalize_adj(self, adj):
        # 对称归一化
        D = torch.diag(torch.pow(adj.sum(1), -0.5))
        return D @ adj @ D
    
    def forward(self, x):
        # x: [B, T, V, C]
        B, T, V, C = x.size()
        
        # 空间图卷积
        x = x.view(B*T, V, C)  # 合并批次和时间维度
        x = self.joint_emb(x)  # [B*T, V, hidden]
        
        # 时间维度卷积
        x = x.view(B, T, V, -1).permute(0, 3, 1, 2)  # [B, hidden, T, V]
        x = x.contiguous().view(B, -1, T)  # [B, hidden*V, T]
        x = self.temporal_conv(x)  # [B, hidden, T]
        
        return x.permute(0, 2, 1)  # [B, T, hidden]

class GCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, adj):
        super().__init__()
        self.adj = nn.Parameter(adj, requires_grad=False)  # 固定骨骼连接
        self.gcn = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # x: [N, V, C]
        x = torch.einsum('nvc, vw->nwc', x, self.adj)  # 图扩散
        x = x.permute(0, 2, 1)  # [N, C, V]
        x = self.gcn(x)        # [N, out_C, V]
        x = self.bn(x)
        x = self.act(x)
        return x.permute(0, 2, 1)  # [N, V, out_C]

#时间维度对齐模块 (新增时间同步层)
class TemporalAlign(nn.Module):
    def __init__(self, feat_dim=512):
        super().__init__()
        self.conv = nn.Conv1d(feat_dim, feat_dim, 3, padding=1)  # 特征平滑卷积
        self.pos_enc = nn.Parameter(torch.zeros(1, 1000, feat_dim))  # 位置编码
        
    def forward(self, x, target_length):
        B, T, D = x.size()
        if T == target_length:
            return x + self.pos_enc[:, :T, :]
        
        # 计算关键帧权重
        mean_feat = x.mean(dim=1, keepdim=True)           # [B,1,D]
        frame_scores = torch.norm(x - mean_feat, p=2, dim=2)  # [B,T]
        
        # 动态选择策略
        if T > target_length:  # 下采样选择关键帧
            # 选择差异最大的target_length帧
            _, top_indices = torch.topk(frame_scores, k=target_length, dim=1)
            sorted_indices, _ = torch.sort(top_indices, dim=1)  # 保持时序连续性
            selected = torch.gather(x, 1, sorted_indices.unsqueeze(-1).expand(-1,-1,D))
        else:  # 上采样情况
            # 选择所有帧并插值
            selected = x.permute(0, 2, 1)
            selected = F.interpolate(selected, size=target_length, mode='nearest')
            selected = selected.permute(0, 2, 1)
        
        # 特征平滑与位置编码
        selected = selected.permute(0, 2, 1)  # [B, D, T]
        selected = self.conv(selected)         # 1D卷积平滑
        return (selected.permute(0, 2, 1) + self.pos_enc[:, :target_length, :]).contiguous()

class DynamicCrossModalLayer(nn.Module):
    def __init__(self, clip_dim=512, num_heads=8):
        super().__init__()
        self.PoseEncoder = PoseEncoderGCN(2, clip_dim)
        self.temporal_align = TemporalAlign(clip_dim)
        self.num_heads = num_heads
        # 多头注意力动态门控
        self.mask_gen = nn.Sequential(
            nn.Linear(2*clip_dim, 4*clip_dim),
            nn.GELU(),
            nn.Linear(4*clip_dim, num_heads),
            nn.Sigmoid()
        )
        
        # 多头注意力参数
        self.q_proj = nn.Linear(clip_dim, clip_dim)
        self.kv_proj = nn.Linear(clip_dim, 2*clip_dim)
        self.mha_dropout = nn.Dropout(0.1)
        
        # 归一化层
        self.norm1 = nn.LayerNorm(clip_dim)
        self.norm2 = nn.LayerNorm(clip_dim)
        
    def forward(self, clip_feat, pose_feat):
        B, T, D = clip_feat.size()
        
        # 姿态特征编码与对齐
        pose_proj = self.PoseEncoder(pose_feat)
        pose_proj = self.temporal_align(pose_proj, T)
        
        # 动态门控生成
        concat_feat = torch.cat([clip_feat, pose_proj], dim=-1)
        head_gates = self.mask_gen(concat_feat)  # [B, T, H]
        
        # 查询/键值投影
        q = self.q_proj(clip_feat).view(B, T, self.num_heads, -1).permute(0,2,1,3)  # [B, H, T, D/H]
        k, v = self.kv_proj(pose_proj).chunk(2, dim=-1)
        k = k.view(B, T, self.num_heads, -1).permute(0,2,1,3)  # [B, H, T, D/H]
        v = v.view(B, T, self.num_heads, -1).permute(0,2,1,3)
        
        # 注意力计算
        attn_weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn_weights = attn_weights * head_gates.permute(0,2,1).unsqueeze(-1)  # 应用门控
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = (attn_weights @ v).permute(0,2,1,3).contiguous().view(B, T, D)
        
        # 残差连接
        attn_output = self.norm1(clip_feat + self.mha_dropout(attn_output))
        return self.norm2(attn_output)

class HierarchicalMMAFusion(nn.Module):
    def __init__(self, num_layers=4, clip_dim=512):
        super().__init__()
        self.layers = nn.ModuleList([
            DynamicCrossModalLayer(clip_dim=clip_dim)
            for _ in range(num_layers)
        ])
        
        # 自适应融合权重
        self.fusion_weights = nn.Parameter(torch.ones(num_layers))
        self.layer_scale = nn.ParameterList([
            nn.Parameter(torch.ones(clip_dim) * 1e-6 )
            for _ in range(num_layers)
        ])
    
    def forward(self, clip_feat, pose_feat):
        all_outputs = []
        current = clip_feat
        
        for i, layer in enumerate(self.layers):
            # 跨层残差连接
            residual = current
            current = layer(current, pose_feat)
            current = residual + self.layer_scale[i] * current
            
            # 记录各层输出
            all_outputs.append(current)
        
        # 自适应加权融合
        weights = F.softmax(self.fusion_weights, dim=0)
        fused_output = sum(w * out for w, out in zip(weights, all_outputs))
        return fused_output

# def test_dimension_flow():
#     B, T, D = 6, 8, 512
#     pose_data = torch.randn(B, 165, 17, 2)
    
#     # 初始化模型
#     model = HierarchicalMMAFusion(num_layers=3)
    
#     # 模拟输入
#     clip_feat = torch.randn(B, T, D)
#     output = model(clip_feat, pose_data)
    
#     # 验证输出维度
#     assert output.shape == (B, T, D), f"输出维度错误，实际维度：{output.shape}"
#     print("测试通过！维度流正常")

# test_dimension_flow()
