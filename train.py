import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import emoji
import re
import random
from sklearn.metrics import f1_score


# 清理表情符号和特殊字符
def clean_text(text):
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？]', '', text)
    return text.strip()


# 自定义 Collate 函数
def custom_collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        if key == 'quadruplets' or key == 'text':
            collated[key] = [item[key] for item in batch]
        else:
            collated[key] = torch.stack([item[key] for item in batch])
    return collated


# 1. NER数据集类
class HateSpeechNERDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128, is_train=True):
        self.is_train = is_train
        self.data = self.load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.group_labels = ['Region', 'Racism', 'Sexism', 'LGBTQ', 'others', 'non-hate']
        self.hate_labels = ['hate', 'non-hate']
        self.ner_labels = ['O', 'B-Target', 'I-Target', 'B-Argument', 'I-Argument']
        self.ner_label2id = {label: idx for idx, label in enumerate(self.ner_labels)}

    def load_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cleaned_data = []
        for item in data:
            # 处理空 content
            if not item.get('content'):
                item['content'] = "空文本"
            else:
                item['content'] = clean_text(item['content'])

            # 处理空或无效 output
            if self.is_train and (not item.get('output') or '[END]' not in item.get('output', '')):
                item['output'] = "NULL | 空文本 | non-hate | non-hate [END]"
            elif self.is_train and item.get('output'):
                # 处理组合标签
                output = item['output']
                parts = output.split(' | ')
                if len(parts) >= 4:
                    targeted_group = parts[2]
                    if ',' in targeted_group:
                        targeted_group = targeted_group.split(',')[0].strip()
                        parts[2] = targeted_group
                        item['output'] = ' | '.join(parts[:-1]) + ' [END]'

            cleaned_data.append(item)

        # 过采样 hate 样本
        if self.is_train:
            hate_data = [item for item in cleaned_data if 'hate' in item.get('output', '')]
            cleaned_data.extend(random.sample(hate_data, min(len(hate_data), 1000)))

        return cleaned_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['content']
        output = item.get('output', '')

        # 清理文本
        cleaned_text = clean_text(text)
        if not cleaned_text:
            cleaned_text = "空文本"

        # 编码文本
        encoding = self.tokenizer(
            cleaned_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )

        if 'offset_mapping' not in encoding:
            print(f"Encoding keys: {encoding.keys()}")
            raise ValueError(f"offset_mapping not found in encoding for text: {cleaned_text}")

        offsets = encoding['offset_mapping'].squeeze().tolist()

        # 解析四元组
        quadruplets = self.parse_output(output) if self.is_train else []

        # 确保至少有一个四元组
        if self.is_train and not quadruplets:
            quadruplets = [{
                'target': 'NULL',
                'argument': cleaned_text[:30],
                'targeted_group': 'non-hate',
                'hateful': 'non-hate'
            }]

        # 生成NER标签
        ner_labels = ['O'] * self.max_length
        if self.is_train:
            for quad in quadruplets[:1]:
                target = quad['target']
                argument = quad['argument']
                for entity, tag in [(target, 'Target'), (argument, 'Argument')]:
                    if entity == 'NULL' or not entity:
                        continue
                    # 使用 offset_mapping 精确匹配
                    entity_start = cleaned_text.find(entity)
                    if entity_start == -1:
                        continue
                    entity_end = entity_start + len(entity)
                    for i, (start, end) in enumerate(offsets):
                        if start >= entity_start and start < entity_end:
                            if start == entity_start:
                                ner_labels[i] = f'B-{tag}'
                            else:
                                ner_labels[i] = f'I-{tag}'

        ner_label_ids = [self.ner_label2id[label] for label in ner_labels]

        # 分类标签
        class_label = 0
        if self.is_train:
            try:
                group_label = self.group_labels.index(quadruplets[0]['targeted_group'])
                hate_label = self.hate_labels.index(quadruplets[0]['hateful'])
                class_label = group_label * 2 + hate_label
            except (IndexError, ValueError) as e:
                print(f"Invalid quadruplets in output: {output}")
                class_label = self.group_labels.index('non-hate') * 2

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'ner_labels': torch.tensor(ner_label_ids),
            'class_label': torch.tensor(class_label),
            'text': cleaned_text,
            'quadruplets': quadruplets[:1] if self.is_train else []
        }

    def parse_output(self, output):
        quadruplets = []
        for quad in output.split(' [SEP] '):
            if quad.strip():
                parts = quad.replace(' [END]', '').split(' | ')
                if len(parts) == 4:
                    targeted_group = parts[2]
                    if ',' in targeted_group:
                        targeted_group = targeted_group.split(',')[0].strip()
                        print(f"Processed combined targeted_group: {parts[2]} -> {targeted_group}")
                    quadruplets.append({
                        'target': parts[0],
                        'argument': parts[1],
                        'targeted_group': targeted_group,
                        'hateful': parts[3]
                    })
        return quadruplets


# 2. 训练函数
def train_model(ner_model, class_model, train_loader, val_loader, device, epochs=3, lr=2e-5):
    ner_optimizer = AdamW(ner_model.parameters(), lr=lr, weight_decay=0.01)
    class_optimizer = AdamW(class_model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    ner_scheduler = get_linear_schedule_with_warmup(ner_optimizer, num_warmup_steps=100, num_training_steps=total_steps)
    class_scheduler = get_linear_schedule_with_warmup(class_optimizer, num_warmup_steps=100,
                                                      num_training_steps=total_steps)

    ner_model.train()
    class_model.train()

    for epoch in range(epochs):
        total_ner_loss = 0
        total_class_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ner_labels = batch['ner_labels'].to(device)
            class_label = batch['class_label'].to(device)

            # NER模型
            ner_outputs = ner_model(input_ids, attention_mask=attention_mask, labels=ner_labels)
            ner_loss = ner_outputs.loss
            ner_optimizer.zero_grad()
            ner_loss.backward()
            torch.nn.utils.clip_grad_norm_(ner_model.parameters(), max_norm=1.0)
            ner_optimizer.step()
            ner_scheduler.step()
            total_ner_loss += ner_loss.item()

            # 分类模型
            class_outputs = class_model(input_ids, attention_mask=attention_mask, labels=class_label)
            class_loss = class_outputs.loss
            class_optimizer.zero_grad()
            class_loss.backward()
            torch.nn.utils.clip_grad_norm_(class_model.parameters(), max_norm=1.0)
            class_optimizer.step()
            class_scheduler.step()
            total_class_loss += class_loss.item()

        # 验证
        ner_model.eval()
        class_model.eval()
        ner_preds_all, ner_labels_all = [], []
        class_preds_all, class_labels_all = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                ner_labels = batch['ner_labels'].to(device)
                class_label = batch['class_label'].to(device)

                ner_outputs = ner_model(input_ids, attention_mask=attention_mask)
                ner_preds = torch.argmax(ner_outputs.logits, dim=-1)
                ner_preds_all.extend(ner_preds.view(-1).cpu().numpy())
                ner_labels_all.extend(ner_labels.view(-1).cpu().numpy())

                class_outputs = class_model(input_ids, attention_mask=attention_mask)
                class_preds = torch.argmax(class_outputs.logits, dim=-1)
                class_preds_all.extend(class_preds.cpu().numpy())
                class_labels_all.extend(class_label.cpu().numpy())

        ner_f1 = f1_score(ner_labels_all, ner_preds_all, average='macro')
        class_f1 = f1_score(class_labels_all, class_preds_all, average='macro')
        print(
            f'Epoch {epoch + 1}, NER Loss: {total_ner_loss / len(train_loader):.4f}, Class Loss: {total_class_loss / len(train_loader):.4f}')
        print(f'NER F1: {ner_f1:.4f}, Class F1: {class_f1:.4f}')

        ner_model.train()
        class_model.train()

    # 保存模型
    ner_model.save_pretrained('ner_model')
    class_model.save_pretrained('class_model')


# 3. 主函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    local_model_path = './pretrained_model/chinese-roberta-wwm-ext'

    tokenizer = BertTokenizerFast.from_pretrained(local_model_path, use_fast=True)
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

    ner_model = BertForTokenClassification.from_pretrained(
        local_model_path,
        num_labels=5
    ).to(device)
    class_model = BertForSequenceClassification.from_pretrained(
        local_model_path,
        num_labels=12
    ).to(device)

    # 加载数据集
    train_data_path = 'data/train.json'

    train_dataset = HateSpeechNERDataset(train_data_path, tokenizer, max_length=128, is_train=True)

    # 拆分训练集和验证集
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)

    # 训练
    train_model(ner_model, class_model, train_loader, val_loader, device, epochs=3)
    print("训练完成！模型已保存到 'ner_model' 和 'class_model'.")

if __name__ == '__main__':
    main()