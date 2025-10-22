import json
import torch
import torch.nn as nn
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 

import time
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn import functional as F
import math

import clip

from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

import sys

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 确保每次写入后都刷新到文件

    def flush(self):
        # 这个flush方法是为了兼容可能调用sys.stdout.flush()的代码
        self.terminal.flush()
        self.log.flush()

# 将标准输出重定向到自定义的Logger
sys.stdout = Logger('/your/path/eval.log')

classnames = ["classnames", "of", "current", "dataset"]

device = torch.device( 
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = LoRAtransformer(clip_model)
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 32
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        print("Initializing a generic context")
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)

        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat(
            [
                prefix, 
                ctx,   
                suffix, 
            ],
            dim=1,
        )
        return prompts

class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        # lora_clip_model = CLIPWithLoRA(clip_model)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

class LoRAModule(torch.nn.Module):
    def __init__(self, d_model, rank, device='cuda'):
        super(LoRAModule, self).__init__()
        self.rank = rank
        self.A = torch.nn.Parameter(torch.randn(d_model, rank).to(device))
        torch.nn.init.normal_(self.A, mean=0.0, std=0.02)
        self.B = torch.nn.Parameter(torch.zeros(rank, d_model).to(device))

    def forward(self):
        # 应用低秩更新
        return self.A @ self.B
    
class ModifiedMultiheadAttention(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.lora_query = LoRAModule(attn.embed_dim, 5, attn.in_proj_weight.device)
        self.lora_key = LoRAModule(attn.embed_dim, 5, attn.in_proj_weight.device)

    def forward(self, query, key, value, need_weights=True, attn_mask=None):
        # LoRA update
        updated_query_weight = self.attn.in_proj_weight[:self.attn.embed_dim, :] + self.lora_query()
        updated_key_weight = self.attn.in_proj_weight[self.attn.embed_dim:2*self.attn.embed_dim, :] + self.lora_key()

        # Using parameters after training
        query = torch.matmul(query, updated_query_weight)
        key = torch.matmul(key, updated_key_weight)
        value = torch.matmul(value, self.attn.in_proj_weight[2*self.attn.embed_dim:, :])  # Value权重保持不变

        # Attention score
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.attn.embed_dim)
        attn = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(attn, value)

        if need_weights:
            return output, attn
        else:
            return output, None

class LoRAtransformer(torch.nn.Module):
    def __init__(self, clip_model, num_lora_blocks=2):
        super(LoRAtransformer, self).__init__()
        self.clip_model = clip_model
        # Wrap attention modules
        for i, block in enumerate(self.clip_model.transformer.resblocks):
            if i < num_lora_blocks:
                block.attn = ModifiedMultiheadAttention(block.attn)
            else:
                break

    def forward(self, x):
        return self.clip_model.transformer(x)