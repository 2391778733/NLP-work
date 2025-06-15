import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, BertForSequenceClassification
from tqdm import tqdm
import emoji
import re


# 清理表情符号和特殊字符
def clean_text(text):
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？]', '', text)
    return text.strip()


# 1. 测试数据集类
class HateSpeechTestDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.data = self.load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['content']
        cleaned_text = clean_text(text)
        if not cleaned_text:
            cleaned_text = "空文本"

        encoding = self.tokenizer(
            cleaned_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        if 'offset_mapping' not in encoding and 'offsets_mapping' not in encoding:
            print(f"Encoding keys: {encoding.keys()}")
            raise ValueError(f"offset_mapping not found in encoding for text: {cleaned_text} (original: {text})")

        # 确保 offsets 是元组列表
        offsets = encoding.get('offset_mapping', encoding.get('offsets_mapping')).squeeze()
        if isinstance(offsets, torch.Tensor):
            offsets = offsets.tolist()
        if not all(isinstance(offset, (tuple, list)) and len(offset) == 2 for offset in offsets):
            print(f"Invalid offsets format for text: {cleaned_text}, offsets: {offsets[:10]}")
            offsets = [(0, 0)] * self.max_length

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'offsets': offsets,
            'text': cleaned_text,
            'id': item['id']
        }


# 2. 提取四元组
def extract_quadruplets(text, offsets, ner_model, class_model, tokenizer, device):
    cleaned_text = clean_text(text)
    if not cleaned_text:
        cleaned_text = "空文本"

    encoding = tokenizer(
        cleaned_text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        return_offsets_mapping=True
    )
    if 'offset_mapping' not in encoding and 'offsets_mapping' not in encoding:
        print(f"Encoding keys: {encoding.keys()}")
        raise ValueError(f"offset_mapping not found in encoding for text: {cleaned_text} (original: {text})")

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # NER预测
    ner_model.eval()
    with torch.no_grad():
        ner_outputs = ner_model(input_ids, attention_mask=attention_mask)
        ner_preds = torch.argmax(ner_outputs.logits, dim=-1)[0].cpu().numpy()

    # 调试：检查 ner_preds 和 offsets 的长度
    if len(ner_preds) != len(offsets):
        print(
            f"Length mismatch for text: {cleaned_text}, ner_preds length: {len(ner_preds)}, offsets length: {len(offsets)}")
        offsets = offsets[:len(ner_preds)]  # 截断以匹配

    # 提取Target和Argument
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    ner_labels = ['O', 'B-Target', 'I-Target', 'B-Argument', 'I-Argument']
    target, argument = [], []
    current_entity = None
    current_entity_tokens = []
    for i, (label_id, offset) in enumerate(zip(ner_preds, offsets)):
        start, end = offset if isinstance(offset, (tuple, list)) else (0, 0)
        if start == 0 and end == 0:  # [CLS], [SEP], [PAD]
            continue
        label = ner_labels[label_id]
        token_text = cleaned_text[start:end] if start < len(cleaned_text) else ''

        if label.startswith('B-'):
            if current_entity and current_entity_tokens:
                if current_entity == 'Target':
                    target.append(''.join(current_entity_tokens))
                else:
                    argument.append(''.join(current_entity_tokens))
            current_entity = label[2:]
            current_entity_tokens = [token_text]
        elif label.startswith('I-') and current_entity == label[2:]:
            current_entity_tokens.append(token_text)
        else:
            if current_entity and current_entity_tokens:
                if current_entity == 'Target':
                    target.append(''.join(current_entity_tokens))
                else:
                    argument.append(''.join(current_entity_tokens))
            current_entity = None
            current_entity_tokens = []

    if current_entity and current_entity_tokens:
        if current_entity == 'Target':
            target.append(''.join(current_entity_tokens))
        else:
            argument.append(''.join(current_entity_tokens))

    # 分类预测
    class_model.eval()
    with torch.no_grad():
        class_outputs = class_model(input_ids, attention_mask=attention_mask)
        class_pred = torch.argmax(class_outputs.logits, dim=-1).item()

    group_labels = ['Region', 'Racism', 'Sexism', 'LGBTQ', 'others', 'non-hate']
    hate_labels = ['hate', 'non-hate']
    group_idx = class_pred // 2
    hate_idx = class_pred % 2
    targeted_group = group_labels[group_idx]
    hateful = hate_labels[hate_idx]

    # 构造四元组
    quadruplets = []
    target = target[0] if target else 'NULL'
    argument = argument[0] if argument else cleaned_text[:10]
    quadruplets.append({
        'target': target,
        'argument': argument,
        'targeted_group': targeted_group,
        'hateful': hateful
    })

    return quadruplets


# 3. 推理函数
def predict(ner_model, class_model, test_loader, tokenizer, device, output_file):
    results = []
    for batch in tqdm(test_loader, desc='Predicting'):
        text = batch['text'][0]
        offsets = batch['offsets']
        try:
            quadruplets = extract_quadruplets(text, offsets, ner_model, class_model, tokenizer, device)
        except Exception as e:
            print(f"Error processing text: {text}, error: {e}")
            quadruplets = [{
                'target': 'NULL',
                'argument': text[:10],
                'targeted_group': 'non-hate',
                'hateful': 'non-hate'
            }]

        # 格式化输出
        output_lines = []
        for quad in quadruplets:
            line = f"{quad['target']} | {quad['argument']} | {quad['targeted_group']} | {quad['hateful']} [END]"
            output_lines.append(line)
        results.append(' [SEP] '.join(output_lines))

    # 保存到TXT文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + '\n')


# 4. 主函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    local_model_path = './pretrained_model/chinese-roberta-wwm-ext'

    tokenizer = BertTokenizerFast.from_pretrained(local_model_path, use_fast=True)
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

    ner_model = BertForTokenClassification.from_pretrained('ner_model').to(device)
    class_model = BertForSequenceClassification.from_pretrained('class_model').to(device)

    # 加载测试数据集
    test_data_path = 'data/test1.json'
    test_dataset = HateSpeechTestDataset(test_data_path, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 推理并保存
    output_file = 'submission.txt'
    predict(ner_model=ner_model, class_model=class_model, test_loader=test_loader, tokenizer=tokenizer, device=device, output_file=output_file)
    print(f'预测已保存到 {output_file}')

if __name__ == '__main__':
    main()