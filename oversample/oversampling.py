import time
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import os
import wandb
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler, BertConfig, \
    RobertaForSequenceClassification, RobertaConfig, RobertaTokenizer
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler, BorderlineSMOTE
from peft import get_peft_model, LoraConfig
from datasets import DatasetDict, load_from_disk
from eda import augument_text
import torch.nn.functional as F
import torch.nn as nn

try:
    data = pd.read_csv('../incidents_train.csv', index_col=0)
except:
    data = pd.read_csv('incidents_train.csv', index_col=0)
# params -------------------------------------------------------------------- #
total_batch_size = 32
try:
    num_gpus = int(os.environ['WORLD_SIZE'])
    print(f"num_gpus:{num_gpus}")
except:
    num_gpus = 1
    print("single gpu")
batch_size = total_batch_size // num_gpus
lr = 5e-5
dropout = 0.0
num_epochs = 3
sample_rate = 0.2
hidden_dropout_prob = dropout
attention_dropout_prob = dropout
lr_schedule = 'cosine_with_warmup'  # cosine
oversample = False
wandb_log = True
sweep = True
lora = False
use_focal_loss = True
label = 'product'
model_name = 'bert'
train_dataset_save_path = './food_dataset_train'
val_dataset_save_path = './food_dataset_val'
aug = True
# --------------------------------------------------------------------------- #
# DDP settings--------------------------------------------------------------- #
backend = 'nccl'
device = 'cuda'
compile = False

ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
torch.manual_seed(1337 + seed_offset)


# --------------------------------------------------------------------------- #
# wandb settings------------------------------------------------------------- #
def broadcast_params(lr, batch_size, num_epochs, dropout):
    # 将lr和batch_size转换为Tensor
    lr_tensor = torch.tensor([lr], dtype=torch.float32).cuda()
    batch_size_tensor = torch.tensor([batch_size], dtype=torch.int32).cuda()
    num_epochs_tensor = torch.tensor([num_epochs], dtype=torch.int32).cuda()
    dropout_tensor = torch.tensor([dropout], dtype=torch.float32).cuda()

    # 使用0号进程广播参数给其他进程
    torch.distributed.broadcast(lr_tensor, src=0)
    torch.distributed.broadcast(batch_size_tensor, src=0)
    torch.distributed.broadcast(num_epochs_tensor, src=0)
    torch.distributed.broadcast(dropout_tensor, src=0)

    # 返回广播后的lr和batch_size
    lr = lr_tensor.item()
    batch_size = batch_size_tensor.item()
    num_epochs = num_epochs_tensor.item()
    dropout = dropout_tensor.item()
    return lr, batch_size, num_epochs, dropout


if wandb_log and master_process:
    wandb.init()
if sweep:
    if master_process:
        wandb_config = wandb.config
        lr = wandb_config.lr
        batch_size = wandb_config.batch_size // num_gpus
        num_epochs = wandb_config.num_epochs
        dropout = wandb_config.dropout
        sample_rate = wandb_config.sample_rate
    lr, batch_size, num_epochs, dropout = broadcast_params(lr, batch_size, num_epochs, dropout)
    print(
        f"Process {ddp_rank}: lr = {lr}, batch_size = {batch_size},num_epochs = {num_epochs},dropout = {dropout},sample_rate={sample_rate}")
# --------------------------------------------------------------------------- #

if model_name == 'bert':
    config = BertConfig.from_pretrained('bert-base-uncased',
                                        hidden_dropout_prob=dropout,
                                        attention_probs_dropout_prob=dropout,
                                        num_labels=len(data[label].unique()))
elif model_name == 'roberta':
    config = RobertaConfig.from_pretrained('roberta-base',
                                           hidden_dropout_prob=dropout,
                                           attention_probs_dropout_prob=dropout,
                                           num_labels=len(data[label].unique())
                                           )


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        alpha: 平衡因子，用于控制正负样本的权重
        gamma: 难易样本的调节因子，gamma 越大，对困难样本的关注度越高
        reduction: 损失聚合方式，'none'、'mean' 或 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: 模型的预测输出，形状为 [batch_size, num_classes]
        targets: 实际标签，形状为 [batch_size]
        """
        # 将输入值进行 softmax，得到类别概率
        probs = F.softmax(inputs, dim=1)
        # 获取目标类别的概率值
        target_probs = probs.gather(1, targets.view(-1, 1)).squeeze(1)

        # 计算 focal loss 的核心公式
        focal_weight = self.alpha * (1 - target_probs) ** self.gamma
        loss = focal_weight * F.cross_entropy(inputs, targets, reduction='none')

        # 根据 reduction 参数选择损失聚合方式
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FoodDataset(Dataset):
    def __init__(self, data, tokenizer, oversample=True):
        # 对所有文本进行分词和编码
        encoded_data = [tokenizer(
            text,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ) for text in (data["title"] + data["text"])]

        # 将编码后的 input_ids 和 attention_mask 提取出来
        input_ids = torch.stack([item['input_ids'].squeeze() for item in encoded_data])
        attention_mask = torch.stack([item['attention_mask'].squeeze() for item in encoded_data])
        labels = torch.tensor(data['label'].values, dtype=torch.long)
        # input_ids = [item['input_ids'].squeeze(0) for item in encoded_data]
        # attention_mask = [item['attention_mask'].squeeze(0) for item in encoded_data]
        # labels = torch.tensor(data['label'].values, dtype=torch.long)
        if oversample:
            # 使用 RandomOverSampler 进行过采样
            label_dist = data['label'].value_counts().to_dict()
            max_count = max(label_dist.values())
            target_sample_count = int(max_count * sample_rate)
            # 创建采样策略，将小于最大样本10%的类别上采样到该数量
            sampling_strategy = {k: target_sample_count for k, v in label_dist.items() if v < target_sample_count}
            ros = RandomOverSampler(sampling_strategy=sampling_strategy)
            # 过采样需要将数据展开成二维数组的形式
            input_ids, labels = ros.fit_resample(input_ids, labels)
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            # input_ids = [torch.tensor(ids, dtype=torch.long) for ids in input_ids]
            # labels = torch.tensor(labels, dtype=torch.long)
            # attention_mask = [(ids != tokenizer.pad_token_id).long() for ids in input_ids]

        # 将过采样后的数据保存到类的属性中
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels.clone()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }


if model_name == 'bert':
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          config=config)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
elif model_name == 'roberta':
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", config=config)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

lora_config = LoraConfig(
    r=16,  # 低秩矩阵的秩
    lora_alpha=16,  # 缩放因子
    lora_dropout=0.1,  # Dropout 概率
    target_modules=["query", "value"],  # LoRA 作用的目标层，通常是注意力层的投影矩阵
)
if lora:
    model = get_peft_model(model, lora_config)
model.to(device)

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# 将labels转换为数值变量
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data[label])

train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
if aug:
    train_df = augument_text(train_df, 'label')
train_dataset = FoodDataset(train_df, tokenizer, oversample=oversample)
test_dataset = FoodDataset(test_df, tokenizer, oversample=False)

# print("start processing dataset")
# if not os.path.exists(train_dataset_save_path) and master_process:
#     os.makedirs(train_dataset_save_path)
#     os.makedirs(val_dataset_save_path)
# try:
#     train_dataset = load_from_disk(train_dataset_save_path)
#     test_dataset = load_from_disk(val_dataset_save_path)
#     print("load dataset from disk")
# except:
#     train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
#     train_dataset = FoodDataset(train_df, tokenizer, oversample=oversample)
#     test_dataset = FoodDataset(test_df, tokenizer, oversample=False)
#     print(train_dataset)
#     if master_process:
#         train_dataset.save_to_disk(train_dataset_save_path)
#         test_dataset.save_to_disk(val_dataset_save_path)
#         print("dataset save to path")


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
if ddp:
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False)

optimizer = AdamW(model.parameters(), lr=lr)
focal_loss = FocalLoss(alpha=1, gamma=2, reduction='mean')

num_training_steps = num_epochs * len(train_loader)

if lr_schedule == 'linear':
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
elif lr_schedule == 'cosine':
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
elif lr_schedule == 'cosine_with_warmup':
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=0.1 * num_training_steps,
        num_training_steps=num_training_steps
    )
else:
    raise ValueError(f'Unrecognized scheduler type: {lr_schedule}')

model.train()

progress_bar = tqdm(range(num_training_steps))

global_step = 0
for epoch in range(num_epochs):
    if ddp:
        train_sampler.set_epoch(epoch)
    for batch in train_loader:
        global_step += 1
        batch = {k: v.to('cuda') for k, v in batch.items()}  # Move batch to GPU if available
        outputs = model(**batch)
        if use_focal_loss:
            loss = focal_loss(outputs.logits, batch["labels"])
        else:
            loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        if wandb_log and master_process:
            wandb.log({
                "step": global_step,
                "train_loss": loss.item(),
                "lr": optimizer.param_groups[0]['lr']
            })
    if master_process:
        model.eval()
        total_predictions = []
        val_loss = 0
        num_batches = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to('cuda') for k, v in batch.items()}  # Move batch to GPU if available
                outputs = model(**batch)
                loss = outputs.loss
                val_loss += loss.item()
                num_batches += 1
                predictions = torch.argmax(outputs.logits, dim=-1)
                total_predictions.extend([p.item() for p in predictions])
        val_loss /= num_batches

        # print(classification_report(test_df.label, total_predictions))
        predicted_labels = label_encoder.inverse_transform(total_predictions)
        gold_labels = label_encoder.inverse_transform(test_df.label.values)
        report = classification_report(gold_labels, predicted_labels, zero_division=0, output_dict=True)
        if wandb_log:
            wandb.log({
                "val_loss": val_loss,
                "val_accuracy": report["accuracy"],
                "classification_report": classification_report(gold_labels, predicted_labels, zero_division=0)
            })
        print(classification_report(gold_labels, predicted_labels, zero_division=0))

# if master_process:
#     model.eval()
#     total_predictions = []
#     val_loss = 0
#     num_batches = 0
#     with torch.no_grad():
#         for batch in test_loader:
#             batch = {k: v.to('cuda') for k, v in batch.items()}  # Move batch to GPU if available
#             outputs = model(**batch)
#             loss = outputs.loss
#             val_loss += loss.item()
#             num_batches += 1
#             predictions = torch.argmax(outputs.logits, dim=-1)
#             total_predictions.extend([p.item() for p in predictions])
#     val_loss /= num_batches
#
#     # print(classification_report(test_df.label, total_predictions))
#     predicted_labels = label_encoder.inverse_transform(total_predictions)
#     gold_labels = label_encoder.inverse_transform(test_df.label.values)
#     report = classification_report(gold_labels, predicted_labels, zero_division=0, output_dict=True)
#     if wandb_log:
#         wandb.log({
#             "val_loss": val_loss,
#             "val_accuracy": report["accuracy"],
#             "classification_report": classification_report(gold_labels, predicted_labels, zero_division=0)
#         })
#     print(classification_report(gold_labels, predicted_labels, zero_division=0))
if ddp:
    destroy_process_group()
