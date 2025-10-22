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

import trimem

import sys

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() 

    def flush(self):
  
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger('/your/path/eval.log')

device = torch.device( 
    "cuda:0" if torch.cuda.is_available() else "cpu"
)


def compute_contrastive_loss(logits):
    # This function computes the contrastive loss given logits computed from the model
    # Diagonal elements are the similarities between corresponding image and text pairs
    labels = torch.arange(logits.size(0)).to(device)
    loss_fct = torch.nn.CrossEntropyLoss()  # This includes log softmax operation
    loss = loss_fct(logits, labels)
    return loss

def compute_accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for
    the specified values of k.

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    Returns:
        list: accuracy at top-k.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res

class EuroSATDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r') as f:
            self.data = [json.loads(line) for line in f]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]['image']
        caption = self.data[idx]['caption']
        try:
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
        except IOError:
            print(f"Error loading image {img_path}! Skipping.")
            return None  # 或者返回一个默认图像

        return image, caption

def count_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

def unfreeze_params(model, block_name):
    # Freeze all parameters in the model initially
    for name, param in model.named_parameters():
        param.requires_grad = False
        print(f"Parameter {name} is frozen.")  # Output the name of the frozen parameters

    # Unfreeze only the prompt embeddings in the prompt learner
    if block_name == "prompt":
        for name, param in model.prompt_learner.named_parameters():
            if "ctx" in name:  # 'ctx' is assumed to be the prompt embeddings
                param.requires_grad = True
                print(f"Parameter {name} is NOT frozen.")  # Confirm that prompt_embeddings parameters are not frozen

    if block_name == "lora":
        for block in model.text_encoder.transformer.clip_model.transformer.resblocks:
            for name, param in block.attn.named_parameters():
                if 'lora' in name:  # Assuming LoRA parameters include 'lora' in their names
                    param.requires_grad = True
                    print(f"Parameter {name} in block {block} is NOT frozen.")  # Confirm LoRA parameters are not frozen

def save_prompt_embeddings(model, save_path):
    
    if hasattr(model.prompt_learner, 'ctx'):
        prompt_embeddings = model.prompt_learner.ctx.data  # Assuming 'ctx' is the prompt embedding parameter
        torch.save(prompt_embeddings, save_path)
        print(f"Prompt_embeddings parameter saved to {save_path}")
    else:
        print("No prompt_embeddings parameter found in prompt_learner!")

def load_prompt_embeddings(model, load_path):
 
    prompt_embeddings = torch.load(load_path)

    if hasattr(model.prompt_learner, 'ctx'):
        with torch.no_grad():  
            model.prompt_learner.ctx.data = prompt_embeddings
            print(f"Prompt_embeddings parameter loaded from {load_path}")
    else:
        print("No prompt_embeddings parameter found in the model!")

def save_lora_weights(model, save_path):
    lora_weights = {}
    # Access LoRA modules through the transformer attribute of the visual component in the CLIP model
    transformer = model.text_encoder.transformer.clip_model.transformer
    for i, block in enumerate(transformer.resblocks):
        if hasattr(block.attn, 'lora_query'):  # Check if LoRA has been applied
            lora_weights[f'block_{i}_query_A'] = block.attn.lora_query.A.data.cpu()
            lora_weights[f'block_{i}_query_B'] = block.attn.lora_query.B.data.cpu()
            lora_weights[f'block_{i}_key_A'] = block.attn.lora_key.A.data.cpu()
            lora_weights[f'block_{i}_key_B'] = block.attn.lora_key.B.data.cpu()

    torch.save(lora_weights, save_path)
    print(f"LoRA weights saved to {save_path}")

def load_lora_weights(model, load_path):
    lora_weights = torch.load(load_path)
    transformer = model.text_encoder.transformer.clip_model.transformer
    for i, block in enumerate(transformer.resblocks):
        if hasattr(block.attn, 'lora_query'):  # Check if LoRA has been applied
            block.attn.lora_query.A.data = lora_weights[f'block_{i}_query_A'].to(block.attn.lora_query.A.device)
            block.attn.lora_query.B.data = lora_weights[f'block_{i}_query_B'].to(block.attn.lora_query.B.device)
            block.attn.lora_key.A.data = lora_weights[f'block_{i}_key_A'].to(block.attn.lora_key.A.device)
            block.attn.lora_key.B.data = lora_weights[f'block_{i}_key_B'].to(block.attn.lora_key.B.device)
    print(f"LoRA weights loaded from {load_path}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])

clip_model, preprocess = clip.load("/opt/data/private/workspace/LLVisMemo/models/ViT-B-32.pt", device=device)
clip_model.float()

model = CustomCLIP(classnames, clip_model).to(device)

save_prompt_path = '/your/path/to/save/prompt.pth'

load_prompt_embeddings(model, save_prompt_path)

save_lora_path = '/your/path/to/save/lora.pth'
save_lora_weights(model, save_lora_path)
load_lora_weights(model, save_lora_path)

# activate prompt
unfreeze_params(model,"None")

dataset = EuroSATDataset(json_file='/your/path/to/save/dataset.json', transform=transform)

total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size 

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Total dataset size: {total_size}")
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

count_trainable_parameters(model)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

model.eval()

start_time = time.time()
# total_loss = 0
accuracy = 0
sum_accuracy = 0
for i, (images, captions) in enumerate(test_loader):
    images = images.to(device)
    captions = list(map(lambda x: x.strip(), captions))
    label_indices = [classnames.index(label) for label in captions]
    label_tensor = torch.tensor(label_indices)
    label_tensor = label_tensor.to(device)
        # Ensure captions are properly stripped of whitespace
    output = model(images)

    accuracy = compute_accuracy(output, label_tensor)
    accuracy = torch.stack(accuracy)
    sum_accuracy += torch.mean(accuracy)
    if (i + 1) % 10 == 0:  # 每10个batch输出一次
        Avg_accuracy = sum_accuracy / 10
        sum_accuracy = 0
        print(f'Acc: {Avg_accuracy:.2f}')

epoch_time = time.time() - start_time


sys.stdout = sys.__stdout__