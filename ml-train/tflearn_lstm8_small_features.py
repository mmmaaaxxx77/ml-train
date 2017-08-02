from __future__ import division, print_function, absolute_import

import pickle

import pyprind
import random
import os
import tflearn
import jieba
import numpy as np
import re
from sklearn import manifold
from tflearn import bidirectional_rnn, BasicLSTMCell
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

jieba.load_userdict("senti/dict/dict.txt.big.txt")
jieba.load_userdict("senti/dict/NameDict_Ch_v2")
jieba.load_userdict("senti/dict/ntusd-negative.txt")
jieba.load_userdict("senti/dict/ntusd-positive.txt")
jieba.load_userdict("senti/dict/鄉民擴充辭典.txt")

dictionary = dict(n=0)
feature_length = 100
dictionary_dist = dict()
X = []
Y = []
default_term_length = 5
max_len = 10


# stop word
"""
stopword_path = "senti/dict/stopwords.json"
stopword_set = set()
with open(stopword_path, 'r') as csv:
    j_list = json.load(csv)
    for word in j_list:
        stopword_set.add(word)
print("讀取stop words")
"""


#comma_tokenizer = lambda x: [y for y in jieba.cut(x, cut_all=False, HMM=True) if y not in stopword_set]
comma_tokenizer = lambda x: jieba.cut(x, cut_all=False, HMM=True)

mode_pkl = "senti/model/seq2seq.pkl"
dictionary_pkl = "senti/model/seq2seq_dict.pkl"

clf = manifold.LocallyLinearEmbedding(feature_length-1, n_components=feature_length, method='standard')


def clear_doc(doc):
    # 去除網址和符號
    r2 = r'https://[a-zA-Z0-9.?/&=:]*'
    r = '[\s！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    r3 = r'\n'
    text = re.sub(r2, '', doc)
    text = re.sub(r, '', text)
    # text = re.sub(r3, ' ', text)
    return text


def set_to_dictionary(term):
    term = str(term)
    if term in dictionary.keys():
        return term
    else:
        _index = len(dictionary)
        dictionary[term] = _index
    return term


def stream_docs(path):
    x = []
    y = []
    with open(path, 'r') as csv:
        for line in csv:
            text = clear_doc(line)
            if len(text) == 0:
                continue
            terms = [set_to_dictionary(i) for i in comma_tokenizer(text)]
            y_part = []
            for ind in range(0, len(terms)):
                if ind+1 < len(terms):
                    y_part.append(terms[ind + 1])
                else:
                    y_part.append("n")
            x.append(terms)
            y.append(y_part)
    return x, y
    # seq_index = np.zeros(len(dictionary), dtype=np.bool)

if not os.path.exists(dictionary_pkl):
    print("處理資料&字典開始")
    doc_streams = [stream_docs(path='senti/data/p/positive.txt')]
        #stream_docs(path='senti/data/n/broken-heart.txt'), stream_docs(path='senti/data/p/adulation.txt'), stream_docs(path='senti/data/p/kindness.txt')]
    print("處理資料&字典完成")
else:
    dictionary = pickle.load(open(dictionary_pkl, 'rb'))


def create_dictionary_dist():
    for key in dictionary.keys():
        _dictionary_array = np.zeros((len(dictionary), ), dtype=np.int8)
        id = dictionary[key]
        _dictionary_array[id] = 1
        dictionary_dist[key] = _dictionary_array

    clf.fit([dictionary_dist[a] for a in dictionary_dist.keys()])
    pyr = pyprind.ProgBar(len(dictionary_dist.keys()))
    for dcst_str in dictionary_dist.keys():
        dictionary_dist[dcst_str] = clf.transform(dictionary_dist[dcst_str])
        pyr.update()


def get_docs_labels(doc_streams):
    docs, y = [], []
    for doc_stream in doc_streams:
        raw_terms, raw_nexts = doc_stream
        matrix = np.zeros((max_len, feature_length), dtype=np.bool)

        il = 0
        while True:
            for it in range(il, len(raw_terms)):
                count = 0
                for word_index in range(0, len(raw_terms[it])):
                    word_str = raw_terms[it][word_index]
                    next_word_str = raw_nexts[it][word_index]
                    matrix[count] = dictionary_dist[word_str]
                    count += 1
                    if count == max_len or word_index == len(raw_terms[it])-1 or\
                            (word_index == len(raw_terms[it])-1 and it == len(raw_terms)-1):
                        docs.append(matrix)
                        matrix = np.zeros((max_len, feature_length), dtype=np.bool)
                        y_m = dictionary_dist[str(next_word_str)][0]
                        y.append(y_m)
                        count = 0
            il += 1
            if il == len(raw_terms):
                break
    return docs, y
print("文字向量建立開始")
create_dictionary_dist()
print("文字向量建立完成")
# Network building
net = tflearn.input_data(shape=[None, max_len, feature_length])
net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128))
#net = tflearn.lstm(net, 512, return_seq=True)
net = tflearn.dropout(net, 0.5)
#net = tflearn.lstm(net, 512, return_seq=True)
#net = tflearn.dropout(net, 0.5)
#net = tflearn.lstm(net, 512)
#net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, feature_length, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
model = tflearn.SequenceGenerator(net, dictionary=dictionary,
                                  seq_maxlen=max_len,
                                  clip_gradients=5.0)

if not os.path.exists(dictionary_pkl):
    #
    print("前處理開始")
    x, y = get_docs_labels(doc_streams)
    print("前處理完成")
    trainX, trainY = x, y
    print(len(dictionary))
    print("{}, {}".format(len(trainX), len(trainY)))
    print("{}, {}".format(np.array(trainX).shape, np.array(trainY).shape))
    model.fit(trainX, trainY, validation_set=0.1, batch_size=128,
              n_epoch=1)
    pickle.dump(dictionary, open(dictionary_pkl, 'wb'), protocol=4)
    #pickle.dump(model, open(mode_pkl, 'wb'), protocol=4)
    model.save(mode_pkl)
else:
    model.load(mode_pkl)

seed = [random.choice(list(dictionary_dist.keys()))]
s = model.generate(5, temperature=0.5, seq_seed=seed)
seed.extend(s[len(seed):])
print("".join(seed))

while True:
    input_str = input("說說話吧: ")

    words = comma_tokenizer(clear_doc(input_str))

    seed = [w for w in words]
    s = model.generate(5, temperature=0.5, seq_seed=seed)
    print("".join(s))

    print("---1.5---")
    s = model.generate(10, temperature=1.5, seq_seed=seed)
    #seed.extend(s[len(seed):])
    print("".join(s))

    print("---1.0---")
    s = model.generate(10, temperature=1.0, seq_seed=seed)
    #seed.extend(s[len(seed):])
    print("".join(s))

    print("---0.5---")
    s = model.generate(10, temperature=0.5, seq_seed=seed)
    #seed.extend(s[len(seed):])
    print("".join(s))

    print("---0.2---")
    s = model.generate(10, temperature=0.2, seq_seed=seed)
    #seed.extend(s[len(seed):])
    print("".join(s))
