# -*- coding: utf-8 -*-
import pandas as pd
import os
import pickle
import jieba
from all_kinds_methods import bm25_evaluation, dpr_evaluation, llm_evaluation, bge_evaluation, xiaobu_evaluation
file_path = './五月线上机构用户问句-标注版-v2.xlsx'
df = pd.read_excel(file_path)
all_results = df[['用户查询', '结果']].dropna(subset=['结果']).values
def read_api_candidates(file_dir):
    all_candidates = []
    all_apis = []
    all_files = os.listdir(file_dir)
    for file in all_files:
        file_path = os.path.join(file_dir, file)
        file_in = open(file_path, 'rb')
        data = pickle.load(file_in)
        for sample in data:
            new_sample = {}
            new_sample['指标名称'] = sample.get('指标名称', '')
            new_sample['指标路径'] = sample.get('指标路径', '')
            new_sample['释义'] = sample.get('释义', '')
            new_sample['参数定义'] = sample.get('参数定义', '')
            all_candidates.append(new_sample)
            all_apis.append(sample['指标名称'])
    return all_candidates, all_apis

def preprocess(all_results, all_candidates, all_apis):
    ground_truth = []
    for item in all_results:
        ground_truth_sample = []
        for api_index in range(len(all_apis)):
            api = all_apis[api_index]
            if api == item[1]:
                ground_truth_sample.append(api_index) # 如果没有对应的api，则ground truth为空，后续就无法命中
        ground_truth.append(ground_truth_sample)
    return [x[0] for x in all_results], all_candidates, ground_truth


all_candidates, all_apis = read_api_candidates('/home/yeqin/basic_docstring')
all_results = [x for x in all_results if len(x[1])>0]
print(len(all_results))
include_in_count = 0
for x in all_results:
    if x[1] in all_apis:
        include_in_count += 1
print(include_in_count/len(all_results))
# upper bound top 1 0.95153

# zero shot retrieval test

qrys, candidates, qc_pairs = preprocess(all_results, all_candidates, all_apis)
r_1, r_5, r_10 = bge_evaluation(qrys, candidates, qc_pairs)
print(r_1, r_5, r_10)
