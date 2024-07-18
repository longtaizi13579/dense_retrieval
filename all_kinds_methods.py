import jieba
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import heapq
import numpy as np
from tqdm import tqdm
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import torch
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagModel
def bm25_evaluation(qrys: list, candidates: dict, qc_pairs: list):
    tokenized_corpus = []
    for candidate in candidates:
        sample = [f'{k}:{v}' for k, v in candidate.items()]
        tokenized_corpus.append(list(jieba.cut('\n'.join(sample))))
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = [list(jieba.cut(sample)) for sample in qrys]
    R_1 = 0
    R_5 = 0
    R_10 = 0
    # 计算BM25得分
    for qry_idx in tqdm(range(len(tokenized_query))):
        scores = bm25.get_scores(tokenized_query[qry_idx])
        add_r_1, add_r_5, add_r_10 = Recall_Calculation(qc_pairs, qry_idx, scores)
        R_1 += add_r_1
        R_5 += add_r_5
        R_10 += add_r_10
    return R_1/len(tokenized_query), R_5/len(tokenized_query), R_10/len(tokenized_query)

def Recall_Calculation(qc_pairs, qry_idx, scores):
    ground_truth = qc_pairs[qry_idx]
    topk_indices_reverse = np.argsort(scores)[-10:]
    topk_indices = topk_indices_reverse[::-1]
    R_1 = 0
    R_5 = 0
    R_10 = 0
    # ground truth为list
    if isinstance(ground_truth, int):
        if ground_truth in topk_indices[0]:
            R_1 = 1
        elif ground_truth in topk_indices[:5]:
            R_1 = 1
            R_5 = 1
        elif ground_truth in topk_indices:
            R_1 = 1
            R_5 = 1
            R_10 = 1
    elif isinstance(ground_truth, list):
        flag_R_1 = 0
        flag_R_5 = 0
        flag_R_10 = 0
        for i in range(len(topk_indices)):
            if topk_indices[i] in ground_truth:
                if i == 0:
                    flag_R_1 += 1
                    flag_R_5 += 1
                    flag_R_10 += 1
                elif i<5:
                    flag_R_5 += 1
                    flag_R_10 += 1
                else:
                    flag_R_10 += 1
        if flag_R_1!= 0 :
            R_1 = 1
        if flag_R_5!= 0 :
            R_5 = 1
        if flag_R_10!= 0 :
            R_10 = 1
    else:
        raise AssertionError()

    return R_1, R_5, R_10


def dpr_evaluation(qrys: list, candidates: dict, qc_pairs: list):
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")
    model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")
    model.cuda()
    model.eval()
    corpus_embedding = []
    with torch.no_grad():
        for candidate in tqdm(candidates):
            #sample = [f'{k}:{v}' for k, v in candidate.items()]
            sample = [f'{k}:{v}' for k, v in candidate.items()]
            input_sentence = '\n'.join(sample)
            input_ids = tokenizer(input_sentence, max_length=512, return_tensors="pt", truncation=True)["input_ids"].cuda()
            embeddings = model(input_ids).pooler_output
            corpus_embedding.append(embeddings)
        corpus_embedding = torch.cat(corpus_embedding, dim=0)
        qry_embedding = []
        for qry in tqdm(qrys):
            input_ids = tokenizer(qry, max_length=512, return_tensors="pt", truncation=True)["input_ids"].cuda()
            embeddings = model(input_ids).pooler_output
            qry_embedding.append(embeddings)
        qry_embedding = torch.cat(qry_embedding, dim=0)
    scores = qry_embedding @ corpus_embedding.T
    R_1 = 0
    R_5 = 0
    R_10 = 0
    # 计算BM25得分
    for qry_idx in tqdm(range(len(scores))):
        score = scores[qry_idx]
        add_r_1, add_r_5, add_r_10 = Recall_Calculation(qc_pairs, qry_idx, score.cpu().numpy())
        R_1 += add_r_1
        R_5 += add_r_5
        R_10 += add_r_10
    return R_1/len(qrys), R_5/len(qrys), R_10/len(qrys)


def llm_evaluation(qrys: list, candidates: dict, qc_pairs: list):
    # model = SentenceTransformer("/home/yeqin/model/gte_Qwen2-1.5B-instruct",trust_remote_code=True)
    model = SentenceTransformer("/home/yeqin/model/gte_Qwen2-7B-instruct",trust_remote_code=True)
    model.max_seq_length = 8192
    model.cuda()
    model.eval()
    corpus = []
    for candidate in candidates:
        sample = [f'{k}:{v}' for k, v in candidate.items()]
        input_sentence = '\n'.join(sample)
        corpus.append(input_sentence)
    with torch.no_grad():
        qry_embedding = model.encode(qrys, prompt_name="query", show_progress_bar=True)
        corpus_embedding = model.encode(corpus, show_progress_bar=True, batch_size=8)
    scores = qry_embedding @ corpus_embedding.T
    R_1 = 0
    R_5 = 0
    R_10 = 0
    # 计算BM25得分
    for qry_idx in tqdm(range(len(scores))):
        score = scores[qry_idx]
        add_r_1, add_r_5, add_r_10 = Recall_Calculation(qc_pairs, qry_idx, score)
        R_1 += add_r_1
        R_5 += add_r_5
        R_10 += add_r_10
    return R_1/len(qrys), R_5/len(qrys), R_10/len(qrys)


# def bge_evaluation(qrys: list, candidates: dict, qc_pairs: list):
#     model = FlagModel('BAAI/bge-large-zh-v1.5', 
#                   query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
#                   use_fp16=True)
#     corpus = []
#     for candidate in candidates:
#         sample = [f'{k}:{v}' for k, v in candidate.items()]
#         input_sentence = '\n'.join(sample)
#         corpus.append(input_sentence)
#     with torch.no_grad():
#         qry_embedding = model.encode_queries(qrys)
#         corpus_embedding = model.encode(corpus)
#     scores = qry_embedding @ corpus_embedding.T
#     R_1 = 0
#     R_5 = 0
#     R_10 = 0
#     # 计算BM25得分
#     for qry_idx in tqdm(range(len(scores))):
#         score = scores[qry_idx]
#         add_r_1, add_r_5, add_r_10 = Recall_Calculation(qc_pairs, qry_idx, score)
#         R_1 += add_r_1
#         R_5 += add_r_5
#         R_10 += add_r_10
#     return R_1/len(qrys), R_5/len(qrys), R_10/len(qrys)



def bge_evaluation(qrys: list, candidates: dict, qc_pairs: list):
    model = FlagModel('BAAI/bge-large-zh-v1.5', 
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True)
    # model = FlagModel('BAAI/bge-large-zh-v1.5', 
    #               use_fp16=True)
    
    # from transformers import AutoModel
    # new_model = AutoModel.from_pretrained('/home/yeqin/code/result')
    # new_model.cuda()
    # model.model = new_model
    corpus = []
    for candidate in candidates:
        sample = [f'{k}:{v}' for k, v in candidate.items()]
        input_sentence = '\n'.join(sample)
        corpus.append(input_sentence)
    with torch.no_grad():
        qry_embedding = model.encode_queries(qrys)
        corpus_embedding = model.encode(corpus)
    scores = qry_embedding @ corpus_embedding.T
    R_1 = 0
    R_5 = 0
    R_10 = 0
    # 计算BM25得分
    for qry_idx in tqdm(range(len(scores))):
        score = scores[qry_idx]
        add_r_1, add_r_5, add_r_10 = Recall_Calculation(qc_pairs, qry_idx, score)
        R_1 += add_r_1
        R_5 += add_r_5
        R_10 += add_r_10
    return R_1/len(qrys), R_5/len(qrys), R_10/len(qrys)


def xiaobu_evaluation(qrys: list, candidates: dict, qc_pairs: list):
    model = SentenceTransformer('/home/yeqin/model/xiaobu-embedding-v2')
    model.cuda()
    model.eval()
    corpus = []
    for candidate in candidates:
        sample = [f'{k}:{v}' for k, v in candidate.items()]
        input_sentence = '\n'.join(sample)
        corpus.append(input_sentence)
    with torch.no_grad():
        qry_embedding = model.encode(qrys, normalize_embeddings=True)
        corpus_embedding = model.encode(corpus, normalize_embeddings=True)
    scores = qry_embedding @ corpus_embedding.T
    R_1 = 0
    R_5 = 0
    R_10 = 0
    # 计算BM25得分
    for qry_idx in tqdm(range(len(scores))):
        score = scores[qry_idx]
        add_r_1, add_r_5, add_r_10 = Recall_Calculation(qc_pairs, qry_idx, score)
        R_1 += add_r_1
        R_5 += add_r_5
        R_10 += add_r_10
    return R_1/len(qrys), R_5/len(qrys), R_10/len(qrys)


def my_bge_evaluation(qrys: list, candidates: dict, qc_pairs: list):
    model = FlagModel('BAAI/bge-large-zh-v1.5', 
                  use_fp16=True)
    
    from transformers import AutoModel
    new_model = AutoModel.from_pretrained('/home/yeqin/code/result')
    new_model.cuda()
    model.model = new_model
    corpus = []
    for candidate in candidates:
        sample = [f'{k}:{v}' for k, v in candidate.items()]
        input_sentence = '\n'.join(sample)
        corpus.append(input_sentence)
    with torch.no_grad():
        qry_embedding = model.encode_queries(qrys)
        corpus_embedding = model.encode(corpus)
    scores = qry_embedding @ corpus_embedding.T
    R_1 = 0
    R_5 = 0
    R_10 = 0
    # 计算BM25得分
    for qry_idx in tqdm(range(len(scores))):
        score = scores[qry_idx]
        add_r_1, add_r_5, add_r_10 = Recall_Calculation(qc_pairs, qry_idx, score)
        R_1 += add_r_1
        R_5 += add_r_5
        R_10 += add_r_10
    return R_1/len(qrys), R_5/len(qrys), R_10/len(qrys)


def xiaobu_evaluation(qrys: list, candidates: dict, qc_pairs: list):
    model = SentenceTransformer('/home/yeqin/model/xiaobu-embedding-v2')
    model.cuda()
    model.eval()
    corpus = []
    for candidate in candidates:
        sample = [f'{k}:{v}' for k, v in candidate.items()]
        input_sentence = '\n'.join(sample)
        corpus.append(input_sentence)
    with torch.no_grad():
        qry_embedding = model.encode(qrys, normalize_embeddings=True)
        corpus_embedding = model.encode(corpus, normalize_embeddings=True)
    scores = qry_embedding @ corpus_embedding.T
    R_1 = 0
    R_5 = 0
    R_10 = 0
    # 计算BM25得分
    for qry_idx in tqdm(range(len(scores))):
        score = scores[qry_idx]
        add_r_1, add_r_5, add_r_10 = Recall_Calculation(qc_pairs, qry_idx, score)
        R_1 += add_r_1
        R_5 += add_r_5
        R_10 += add_r_10
    return R_1/len(qrys), R_5/len(qrys), R_10/len(qrys)