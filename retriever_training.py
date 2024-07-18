from transformers import (
    AdamW,
    HfArgumentParser,
    get_scheduler,
)
from arguments import ModelArguments, DataTrainingArguments, TrainingArguments
import os
import pickle
import torch
import pandas as pd
import re
from collections import defaultdict
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import numpy as np
from FlagEmbedding import FlagModel
from torch.utils.data import DataLoader
import wandb
import logging
import random
import torch.nn.functional as F
def read_api_candidates(file_dir):
    all_candidates = []
    all_apis = []
    identifiers_index = {}
    name_to_identifiers_id = defaultdict(list)
    all_files = os.listdir(file_dir)
    global_count = 0
    for file in all_files:
        file_path = os.path.join(file_dir, file)
        file_in = open(file_path, 'rb')
        data = pickle.load(file_in)
        for sample_index in range(len(data)):
            sample = data[sample_index]
            new_sample = {}
            new_sample['指标名称'] = sample.get('指标名称', '')
            new_sample['指标路径'] = sample.get('指标路径', '')
            new_sample['释义'] = sample.get('释义', '')
            new_sample['参数定义'] = sample.get('参数定义', '')
            # 正则匹配 "()"
            index_formulas = sample.get('指标公式', '')
            bracket  = re.search(r'\(.+?\)', index_formulas).group(0)
            identifiers = eval(bracket.split(',')[0][1:])
            identifiers_index[identifiers] = new_sample
            name_to_identifiers_id[new_sample['指标名称']].append(global_count)
            all_candidates.append(new_sample)
            all_apis.append(sample['指标名称'])
            global_count += 1
    return all_candidates, all_apis, identifiers_index, name_to_identifiers_id


def hard_negative_sample(scores, false_negatives, all_candidates):
    hard_negatives = []
    for qry_idx in tqdm(range(len(scores))):
        score = scores[qry_idx]
        now_hard_negative = []
        topk_indices_reverse = np.argsort(score)[-200:]
        topk_indices = topk_indices_reverse[::-1]
        ptr = 0
        while len(now_hard_negative) != 10:
            index = topk_indices[ptr]
            if index not in false_negatives[qry_idx]:
                now_hard_negative.append(dict_sample_to_string(all_candidates[index]))
            ptr += 1
            if ptr == 200:
                break
        hard_negatives.append(now_hard_negative)
    return hard_negatives

def dict_sample_to_string(candidate):
    sample = [f'{k}:{v}' for k, v in candidate.items()]
    input_sentence = '\n'.join(sample)
    return input_sentence

def build_train_data(file_path, model_name_or_path, tokenizer, identifiers_index, name_to_identifiers_id, all_candidates, data_args):
    training_data = torch.load(file_path)
    model = FlagModel(model_name_or_path, 
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True)
    ground_truth = []
    user_querys = []
    false_negatives = []
    for sample in training_data:
        identify = sample['label']
        if identify in identifiers_index:
            candidate = identifiers_index[identify]
            input_sentence = dict_sample_to_string(candidate)
            ground_truth.append(input_sentence)
            user_querys.append('为这个句子生成表示以用于检索相关文章：'+sample['query'])
            name = identifiers_index[identify]['指标名称']
            false_negatives.append(name_to_identifiers_id[name])
    corpus = []
    for candidate in all_candidates:
        sample = [f'{k}:{v}' for k, v in candidate.items()]
        input_sentence = '\n'.join(sample)
        corpus.append(input_sentence)
    with torch.no_grad():
        qry_embedding = model.encode_queries(user_querys, convert_to_numpy=False)
        corpus_embedding = model.encode(corpus, convert_to_numpy=False)
    scores = qry_embedding @ corpus_embedding.T
    scores = scores.cpu().numpy()
    hard_negatives = hard_negative_sample(scores, false_negatives, all_candidates)
    all_samples = {
        'querys': user_querys,
        'hard_negatives': hard_negatives,
        'ground_truth': ground_truth
    }
    dataset = Dataset.from_dict(all_samples)
    dataset = dataset.shuffle(seed=42)
    def tokenize(examples):
        inputs_querys = tokenizer(
                            examples['querys'],
                            padding=True,
                            truncation=True,
                            return_tensors='pt',
                            max_length=data_args.max_length,
                        )
        batch_hard_negatives = []
        for sample_hard_negatives in examples['hard_negatives']:
            batch_hard_negatives.append(random.choice(sample_hard_negatives))
        hard_negatives = tokenizer(
                            batch_hard_negatives,
                            padding=True,
                            truncation=True,
                            return_tensors='pt',
                            max_length=data_args.max_length,
                        )
        ground_truth = tokenizer(
                            examples['ground_truth'],
                            padding=True,
                            truncation=True,
                            return_tensors='pt',
                            max_length=data_args.max_length,
                        )
        return {
            'querys_ids': inputs_querys['input_ids'],
            'query_attention_mask': inputs_querys['attention_mask'],
            'hard_negatives_ids': hard_negatives['input_ids'],
            'hard_negatives_attention_mask': hard_negatives['attention_mask'],
            'ground_truth_ids': ground_truth['input_ids'],
            'ground_truth_attention_mask': ground_truth['attention_mask'],
        }

    encode_ds = dataset.map(tokenize, batched=True, batch_size=training_args.train_batch_size)
    encode_ds.set_format(type='torch', columns=['querys_ids', 'query_attention_mask', 'hard_negatives_ids',
                                                'hard_negatives_attention_mask', 'ground_truth_ids',
                                                 'ground_truth_attention_mask'])
    dataloader = DataLoader(encode_ds, batch_size=training_args.train_batch_size)# 这里不能设置shuffle，因为dataset.map已经进行了tensor的padding，不同batch之间的shape不一致
    return dataloader
    
        
def retriever_train(model, DataLoader, training_args, logger):
    optimizer = AdamW(model.parameters(),
                weight_decay = training_args.weight_decay,
                lr = training_args.lr)
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.num_warmup_steps,
        num_training_steps=training_args.num_epochs*len(DataLoader),
    )
    torch.cuda.set_device(training_args.local_rank)
    for epoch in tqdm(range(training_args.train_epoch)):
        model.train()
        for idx, batch in enumerate(tqdm(DataLoader, desc=f'Epoch: {epoch+1}')):
            batch = {k:i.cuda() for k, i in batch.items()}
            positive = torch.tensor(list(range(len(batch['querys_ids'])))).cuda()
            qry_embedding = model(batch['querys_ids'], attention_mask=batch['query_attention_mask'], return_dict=True).last_hidden_state[:, 0]
            truth_embedding = model(batch['ground_truth_ids'], attention_mask=batch['ground_truth_attention_mask'], return_dict=True).last_hidden_state[:, 0]
            negatives_embedding = model(batch['hard_negatives_ids'], attention_mask=batch['hard_negatives_attention_mask'], return_dict=True).last_hidden_state[:, 0]
            whole_embedding = torch.cat([truth_embedding, negatives_embedding], dim=0)
            dot_products = torch.matmul(qry_embedding, whole_embedding.T)
            probs = F.log_softmax(dot_products, dim=1)
            loss = F.nll_loss(probs, positive.long())
            if training_args.local_rank == 0:
                logger.info(f'Epoch: {epoch+1}, Batch:{idx+1}, Loss: {loss}')
                wandb.log({"loss": loss})
            loss.backward()
            # 梯度裁剪
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        model.save_pretrained('./result')




if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    wandb.login(key='a7886eb97efa2e436126c244213b8df88c67ffd7')
    wandb.init(project="10JQKA_retriever", name="10JQKA_retriever")
    logger = logging.getLogger(name='my_logger')
    logging.basicConfig(filename=os.path.join('./logger', 'retriever.log'), level=logging.INFO, 
                        format='%(name)s - %(levelname)s - %(message)s')
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModel.from_pretrained(model_args.model_name_or_path)
    model.cuda()
    model.eval()
    all_candidates, all_apis, identifiers_index, name_to_identifiers_id = read_api_candidates(data_args.doc_path)
    data_loader = build_train_data(data_args.file_path, model_args.model_name_or_path, tokenizer, identifiers_index, name_to_identifiers_id, all_candidates, data_args)
    retriever_train(model, data_loader, training_args, logger)