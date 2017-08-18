import json
import pickle
from statistics import mode
import pyprind
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

tStart = time.time()


def print_time():
    tEnd = time.time()
    print("It cost {:0.2f} sec".format(tEnd - tStart))

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

#comma_tokenizer = lambda x: [y for y in jieba.cut(x, cut_all=False, HMM=True) if y not in stopword_set]
#comma_tokenizer = lambda x: [y for y in jieba.cut(x, cut_all=False, HMM=True) if y in p_n_word_set]
comma_tokenizer = lambda x: jieba.cut(x, cut_all=False, HMM=True)
#comma_tokenizer = lambda x: [y for y in x]

max_length = 10

dictionary_length = 2**15
vect = FeatureHasher(n_features=dictionary_length, non_negative=True)

model_pkl = "model/mode_senti3_model.pkl"


def clear_doc(doc):
    # 去除網址和符號
    r2 = r'https://[a-zA-Z0-9.?/&=:]*'
    r = '[\s！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    r3 = r'\n'
    text = re.sub(r2, '', doc)
    text = re.sub(r, '', text)
    # text = re.sub(r3, ' ', text)
    return text


def transform_term_to_id(sentence):
    terms = comma_tokenizer(sentence)
    terms = [{t: 1} for t in terms]
    ids_sci = vect.transform(terms)
    return np.asarray([item.nonzero()[1][0] for item in ids_sci])


def stream_docs(path, label):
    x = []
    y = []

    lines_count = 0
    with open(path, 'r') as file:
        lines_count = sum(1 for _ in file)

    print("{}, {}".format(lines_count, path))

    with open(path, 'r') as file:
        pbar = pyprind.ProgBar(lines_count)
        for line in file:
            pbar.update()
            text = clear_doc(line)
            if len(text) == 0:
                continue
            idterms = transform_term_to_id(text)

            if np.count_nonzero(idterms) < max_length:
                continue

            count = 0
            sub = np.zeros((max_length,), dtype=np.int)
            for i in range(0, idterms.size):
                sub.itemset(count, idterms[i])
                count += 1
                if count == max_length:
                    x.append(sub)
                    y.append(label)
                    count = 0
                    sub = np.zeros((max_length,), dtype=np.int)
                elif i == idterms.size:
                    x.append(sub)
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
            print("ERROR")
    return docs, y

training = True

if training:

    print_time()
    print("讀取&處理資料集")
    doc_streams = [stream_docs(path='data/n/train_n.txt', label=0), stream_docs(path='data/p/train_p.txt', label=1)]
    #test_doc_streams = [stream_docs(path='data/p/kindness.txt', label=1), stream_docs(path='data/n/negative.txt', label=0)]
    print_time()
    print("整理XY開始")
    x, y = get_docs_labels(doc_streams)
    print("前處理完成")

    x_train, y_train = x, y
    y_train = to_categorical(y_train, nb_classes=2)
    print("訓練集切割完成")
    print_time()

# Network building
net = tflearn.input_data(shape=[None, max_length])
net = embedding(net, input_dim=dictionary_length, output_dim=4096)
net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128), return_seq=True)
net = tflearn.dropout(net, 0.5)
net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128), return_seq=True)
net = tflearn.dropout(net, 0.5)
net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128), return_seq=True)
net = tflearn.dropout(net, 0.5)
net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128), return_seq=True)
net = tflearn.dropout(net, 0.5)
net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128), return_seq=True)
net = tflearn.dropout(net, 0.5)
net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128), return_seq=True)
net = tflearn.dropout(net, 0.5)
net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128), return_seq=True)
net = tflearn.dropout(net, 0.5)
net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128), return_seq=True)
net = tflearn.dropout(net, 0.5)
net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128), return_seq=True)
net = tflearn.dropout(net, 0.5)
net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128))
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')
print("網路定義完成")

# Test
#test_x, test_y = get_docs_labels(test_doc_streams)
#test_y = to_categorical(test_y, nb_classes=2)
# Training
if training:
    print("{}, {}".format(np.array(x_train).shape, np.array(y_train).shape))
model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir="log/")


class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self, val_acc_thresh):
        """ Note: We are free to define our init function however we please. """
        self.val_acc_thresh = val_acc_thresh

    def on_epoch_end(self, training_state):
        """ """
        # Apparently this can happen.
        if training_state.val_acc is None: return
        if training_state.val_acc > self.val_acc_thresh:
            raise StopIteration

if training:
    early_stopping_cb = EarlyStoppingCallback(val_acc_thresh=0.82)
    model.fit(x_train, y_train, validation_set=0.2, show_metric=True,
              batch_size=128, run_id="lstm_senti_2t23", n_epoch=100)


def pre_clear(sentence):
    x = []
    text = clear_doc(sentence)
    if len(text) == 0:
        return x
    idterms = transform_term_to_id(text)

    if np.count_nonzero(idterms) < max_length:
        return x

    count = 0
    sub = np.zeros((max_length,), dtype=np.int)
    for i in range(0, idterms.size):
        sub.itemset(count, idterms[i])
        count += 1
        if count == max_length:
            x.append(sub)
            count = 0
            sub = np.zeros((max_length,), dtype=np.int)
    return x

if training:
    model.save(model_pkl)
else:
    model.load(model_pkl)

res_list = []
for seq in pre_clear("從高中以來也認識好多個年頭了 真的很感謝一路相伴的那些好友們 我們永遠都相互扶持彼此打氣 在任何一個流淚的時刻 我知道你們一直在那裏 兩次的心碎都有願意陪在我身邊的人 真的很感謝你們 有你們在 我知道大風大浪都會過去的~~ 當我遇見真正的幸福 你們也會笑著祝福我 感謝你們一直以來的好 希望我們白髮蒼蒼的時候 也能三不五時這樣子聚會聊天玩鬧喔^^ 最愛你們大家了~"):
    test_result = model.predict([seq])
    #print(test_result)
    res_list.append(test_result[0])
    print([a for a in test_result[0]])
print(Counter(np.argmax(res_list, axis=1).tolist()))


print("訓練SGD ...")
n_train_file = "data/n/train_n.txt"
p_train_file = "data/p/train_p.txt"
sgd_max_length = 100
sgd_model_pkl = "model/sgd2.pkl"


def sgd_pre_clear(line):
    sub_x = [0] * sgd_max_length
    count = 0
    #if len(pre_clear(line)) < sgd_max_length:
    #    return None, None
    for seq in pre_clear(line):
        predict_result = model.predict([seq])[0]
        if count == sgd_max_length:
            break
        sub_x[count] = 1 if predict_result[0] < predict_result[1] else 0
        count += 1
    return sub_x, mode(sub_x)

clf = None
if training:
    train_x, train_y = [], []
    with open(n_train_file, 'r') as csv:
        for line in csv:
            sub_x, y_label = sgd_pre_clear(line)
            if sub_x is not None:
                train_x.append(sub_x)
                train_y.append(0)
    with open(p_train_file, 'r') as csv:
        for line in csv:
            sub_x, y_label = sgd_pre_clear(line)
            if sub_x is not None:
                train_x.append(sub_x)
                train_y.append(1)

    print("{}, {}".format(np.asarray(train_x).shape, np.asarray(train_y).shape))

    c = list(zip(train_x, train_y))
    random.shuffle(c)
    train_x, train_y = zip(*c)

    print("訓練SGD前處理完成")
    #train_y = to_categorical(train_y, nb_classes=2)
    #print("{}, {}".format(np.asarray(train_x).shape, train_y))
    clf = SGDClassifier(loss="modified_huber")
    clf.fit(train_x, train_y)

    joblib.dump(clf, sgd_model_pkl)

    print(clf.score(train_x, train_y))
    print("訓練SGD end ...")
else:
    clf = joblib.load(sgd_model_pkl)

while True:
    input_str = input("說說話吧: ")
    seqs = []
    sub_x, y_label = sgd_pre_clear(input_str)
    if sub_x is not None:
        result = clf.predict([sub_x])
        result_proba = clf.predict_proba([sub_x])
        print(result)
        print(result_proba)
