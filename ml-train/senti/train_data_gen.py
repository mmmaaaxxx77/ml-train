import json
import pickle
from statistics import mode

import jieba
import numpy as np
import os
import random
import re
import tflearn
import time
from collections import Counter
from gensim.models import word2vec
from sklearn.externals import joblib
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from tflearn import embedding, bidirectional_rnn, BasicLSTMCell
from tflearn.data_utils import pad_sequences, to_categorical

jieba.load_userdict("dict/dict.txt.big.txt")
jieba.load_userdict("dict/NameDict_Ch_v2")
jieba.load_userdict("dict/ntusd-negative.txt")
jieba.load_userdict("dict/ntusd-positive.txt")
jieba.load_userdict("dict/鄉民擴充辭典.txt")

print("--- 測試斷詞 ---")
print('/'.join(jieba.cut('柯P是柯市長', HMM=False)))
print("--- 測試斷詞 end ---")

comma_tokenizer = lambda x: jieba.cut(x, cut_all=False, HMM=True)

n_file = ""
p_file = ""
valve = 30

total = 100


def clear_doc(doc):
    # 去除網址和符號
    r2 = r'https://[a-zA-Z0-9.?/&=:]*'
    r = '[\s！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    r3 = r'\n'
    text = re.sub(r2, '', doc)
    text = re.sub(r, '', text)
    # text = re.sub(r3, ' ', text)
    return text


def stream_docs(path):
    x = []
    with open(path, 'r') as csv:
        for line in csv:
            text = clear_doc(line)
            text_seq = comma_tokenizer(text)
            text = [seq for seq in text_seq]
            if len(text) > valve:
                x.append(line)
    return x

print("處理檔案")
doc_streams_n = [stream_docs(path='data/n/twneg.txt'),
                 stream_docs(path='data/n/negative.txt'),
                 stream_docs(path='data/n/broken-heart.txt'),
                 stream_docs(path='data/n/neg.test')]
doc_streams_p = [stream_docs(path='data/p/kindness.txt'),
                 stream_docs(path='data/p/twpos.txt'),
                 stream_docs(path='data/p/adulation.txt'),
                 stream_docs(path='data/p/positive.txt'),
                 stream_docs(path='data/p/pos.test')]


def gen_train_data(doc_streams, path):
    count = 0
    x = []
    for doc in doc_streams:
        for sentence in doc:
            x.append(sentence)
    random.shuffle(x)

    if os.path.exists(path):
        os.remove(path)

    with open(path, 'a') as file:
        for sentence in x[0:total]:
            file.write(sentence)
            count += 1
    print("量: {}, {}".format(count, path))

print("開始建立")
gen_train_data(doc_streams_n, "data/n/train_n.txt")
print("n 建立完成")
gen_train_data(doc_streams_p, "data/p/train_p.txt")
print("p 建立完成")
