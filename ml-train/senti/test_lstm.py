import json
import pickle

import jieba
import numpy as np
import os
import re
import tflearn
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from tflearn.data_utils import pad_sequences, to_categorical

jieba.load_userdict("dict/dict.txt.big.txt")
jieba.load_userdict("dict/NameDict_Ch_v2")
jieba.load_userdict("dict/ntusd-negative.txt")
jieba.load_userdict("dict/ntusd-positive.txt")
jieba.load_userdict("dict/鄉民擴充辭典.txt")

print("--- 測試斷詞 ---")
print('/'.join(jieba.cut('柯P是柯市長', HMM=False)))
print("--- 測試斷詞 end ---")

# stop word
stopword_path = "dict/stopwords.json"
stopword_set = set()
with open(stopword_path, 'r') as csv:
    j_list = json.load(csv)
    for word in j_list:
        stopword_set.add(word)
print("讀取stop words")

# 情緒字詞
p_word_path = "dict/ntusd-positive.txt"
p_word_set = set()
with open(p_word_path, 'r') as csv:
    for line in csv:
        p_word_set.add(line)

n_word_path = "dict/ntusd-negative.txt"
n_word_set = set()
with open(n_word_path, 'r') as csv:
    for line in csv:
        n_word_set.add(line)

p_n_word_set = p_word_set.union(n_word_set)
print("讀取情緒字詞")

comma_tokenizer = lambda x: [y for y in jieba.cut(x, cut_all=False, HMM=True) if y not in stopword_set]
#comma_tokenizer = lambda x: [y for y in jieba.cut(x, cut_all=False, HMM=True) if y in p_n_word_set]
#comma_tokenizer = lambda x: jieba.cut(x, cut_all=False, HMM=True)
#comma_tokenizer = lambda x: [y for y in x]

dictionary = set()
dictionary_dict = None


def clear_doc(doc):
    # 去除網址和符號
    r2 = r'https://[a-zA-Z0-9.?/&=:]*'
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    r3 = r'\n'
    text = re.sub(r2, '', doc)
    text = re.sub(r, '', text)
    # text = re.sub(r3, ' ', text)
    return text


def set_to_dictionary(term):
    dictionary.add(term)
    return term


def stream_docs(path, label):
    x = []
    y = []
    with open(path, 'r') as csv:
        for line in csv:
            text = clear_doc(line)
            if len(text) == 0:
                continue
            x.append([set_to_dictionary(i) for i in comma_tokenizer(text)])
            y.append(label)
    return x, y


def get_docs_labels(doc_streams):
    docs, y = [], []
    for doc_stream in doc_streams:
        try:
            t_x, t_y = doc_stream
            docs.extend(t_x)
            y.extend(t_y)
        except Exception:
            return None, None
    return docs, y

doc_streams = [stream_docs(path='data/n/broken-heart.txt', label=0), stream_docs(path='data/p/adulation.txt', label=1),
               stream_docs(path='data/p/kindness.txt', label=1)]

documents, labels = get_docs_labels(doc_streams)
default_word_count = 2 ** 10
print("讀取檔案完成")

# hash table
list_dictionary = list(dictionary)
dictionary_dict = {list_dictionary[v]: v for v in range(0, len(dictionary))}
print("處理hash table")


def to_x_y(documents, labels):
    xx = []
    yy = []
    for i in range(0, len(documents)):
        document = documents[i]
        if len(document) != 0:
            document_x = np.zeros((default_word_count, ), dtype=np.float)
            for term in document:
                index = dictionary_dict[term]
                if index < default_word_count:
                    document_x[index] += 1
                xx.append([document_x])
                yy.append(labels[i])
    return xx, yy

x, y = to_x_y(documents, labels)
print("前處理完成")

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train, y_train = x, y
# Converting labels to binary vectors
y_train = to_categorical(y_train, nb_classes=default_word_count)
print("訓練集切割完成")

# Network building
net = tflearn.input_data(shape=[None, 1, default_word_count])
net = tflearn.lstm(net, 128, return_seq=True)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, default_word_count, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')
print("網路定義完成")

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
print(len(x_train[0][0]))
model.fit(x_train, y_train, validation_set=0.2, show_metric=True,
          batch_size=100)
