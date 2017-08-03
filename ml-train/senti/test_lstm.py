import json
import pickle

import jieba
import numpy as np
import os
import re
import tflearn
from gensim.models import word2vec
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

dictionary = dict()
dict_counter = 0
dictionary_dict = None
max_length = 20

model_pkl = "model/mode_senti3_model.pkl"
dictionary_pkl = "model/mode_senti3_dictionary.pkl"


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
    if term in dictionary.keys():
        return dictionary[term]
    else:
        _index = len(dictionary)
        dictionary[term] = _index
    return _index


def stream_docs(path, label):
    x = []
    y = []
    with open(path, 'r') as csv:
        for line in csv:
            text = clear_doc(line)
            if len(text) == 0:
                continue
            #x.append([set_to_dictionary(i) for i in comma_tokenizer(text)])
            terms = comma_tokenizer(text)
            terms = list(terms)
            if len(terms) < max_length:
                continue
            count = 0
            sub = [0]*max_length
            for i in range(0, len(terms)):
                sub[count] = set_to_dictionary(str(terms[i]))
                count += 1
                if count == max_length:
                    x.append(sub)
                    y.append(label)
                    count = 0
                    sub = [0] * max_length
    return x, y


def test_stream_docs(path, label):
    x = []
    y = []
    with open(path, 'r') as csv:
        for line in csv:
            text = clear_doc(line)
            if len(text) == 0:
                continue
            terms = comma_tokenizer(text)
            terms = list(terms)
            if len(terms) < max_length:
                continue
            count = 0
            sub = [0]*max_length
            for i in range(0, len(terms)):
                try:
                    sub[count] = dictionary[str(terms[i])]
                except:
                    continue
                count += 1
                if count == max_length:
                    x.append(sub)
                    y.append(label)
                    count = 0
                    sub = [0] * max_length
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

doc_streams = [stream_docs(path='data/n/twneg.txt', label=0), stream_docs(path='data/p/twpos.txt', label=1)]
               #stream_docs(path='data/p/kindness.txt', label=1)]
test_doc_streams = [test_stream_docs(path='data/p/kindness.txt', label=1), test_stream_docs(path='data/n/negative.txt', label=0)]
x, y = get_docs_labels(doc_streams)
print("前處理完成")

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train, y_train = x, y
# Converting labels to binary vectors
y_train = to_categorical(y_train, nb_classes=2)
print("訓練集切割完成")

# Network building
net = tflearn.input_data(shape=[None, max_length])
net = embedding(net, input_dim=len(dictionary), output_dim=1024)
net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128))
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')
print("網路定義完成")

# Test
test_x, test_y = get_docs_labels(test_doc_streams)
test_y = to_categorical(test_y, nb_classes=2)
# Training
print("{}, {}".format(np.array(x_train).shape, np.array(y_train).shape))
model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir="log/")

model.fit(x_train, y_train, validation_set=0.2, show_metric=True,
          batch_size=128, run_id="lstm_senti_2t2", n_epoch=3)


def pre_clear(sentence):
    text = clear_doc(sentence)
    if len(text) == 0:
        return None
    terms = comma_tokenizer(text)
    terms = list(terms)
    if len(terms) < max_length:
        return None
    count = 0
    sub = [0] * max_length
    for i in range(0, len(terms)):
        try:
            sub[count] = dictionary[str(terms[i])]
        except:
            continue
        count += 1
        if count == max_length:
            x.append(sub)
            count = 0
            sub = [0] * max_length
    return sub
model.save(model_pkl)
pickle.dump(dictionary, open(dictionary_pkl, 'wb'), protocol=4)

#model.load(model_pkl)
#dictionary = pickle.load(open(dictionary_pkl, 'rb'))

print(model.predict([pre_clear("從高中以來也認識好多個年頭了 真的很感謝一路相伴的那些好友們 我們永遠都相互扶持彼此打氣 在任何一個流淚的時刻 我知道你們一直在那裏 兩次的心碎都有願意陪在我身邊的人 真的很感謝你們 有你們在 我知道大風大浪都會過去的~~ 當我遇見真正的幸福 你們也會笑著祝福我 感謝你們一直以來的好 希望我們白髮蒼蒼的時候 也能三不五時這樣子聚會聊天玩鬧喔^^ 最愛你們大家了~")]))

while True:
    input_str = input("說說話吧: ")
    print(model.predict(
        [pre_clear(input_str)]))
