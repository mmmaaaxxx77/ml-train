import json
import pickle

import jieba
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
import re
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

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


#comma_tokenizer = lambda x: [y for y in jieba.cut(x, cut_all=False, HMM=True) if y not in stopword_set]
#comma_tokenizer = lambda x: [y for y in jieba.cut(x, cut_all=False, HMM=True) if y in p_n_word_set]
#comma_tokenizer = lambda x: jieba.cut(x, cut_all=False, HMM=True)
comma_tokenizer = lambda x: [y for y in x]

vect = HashingVectorizer(decode_error='ignore',
                         n_features=2 ** 21,
                         tokenizer=comma_tokenizer,
                         non_negative=True)

mode_pkl = "model/text1.pkl"


def clear_doc(doc):
    # 去除網址和符號
    r2 = r'https://[a-zA-Z0-9.?/&=:]*'
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    r3 = r'\n'
    text = re.sub(r2, '', doc)
    text = re.sub(r, '', text)
    # text = re.sub(r3, ' ', text)
    return text


def stream_docs(path, label):
    x = []
    y = []
    with open(path, 'r') as csv:
        for line in csv:
            text = clear_doc(line)
            if len(text) == 0:
                continue
            x.append(text)
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

if os.path.exists(mode_pkl):
    svm = pickle.load(open(mode_pkl, 'rb'))
    print("讀取pkl完成")
else:
    x, y = get_docs_labels(doc_streams)

    x = vect.transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    print("資料前處理完成")
    svm = SVC(kernel='rbf', C=10.0, random_state=0, gamma=0.01, verbose=True)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)
    print("{:.2f}".format(accuracy_score(y_test, y_pred)))

if not os.path.exists(mode_pkl):
    pickle.dump(svm, open(mode_pkl, 'wb'), protocol=4)

# 測試資料
#test_path = [stream_docs(path='data/n/negative.txt', label=0), stream_docs(path='data/p/positive.txt', label=1)]
#test_path = [stream_docs(path='data/n/neg.test', label=0), stream_docs(path='data/p/pos.test', label=1)]
test_path = [stream_docs(path='data/n/twneg.txt', label=0), stream_docs(path='data/p/twpos.txt', label=1)]
x, y = get_docs_labels(test_path)
x = vect.transform(x)
y_pred = svm.predict(x)
print("---小英粉絲團---")
print("{:.2f}".format(accuracy_score(y, y_pred)))

#
while True:
    input_str = input("說說話吧: ")
    input_str = clear_doc(input_str)
    input_str = vect.transform([input_str])
    print(svm.predict(input_str))
    pre = svm.predict(input_str)[0]
    source = svm.score(input_str, [pre])

    print("{}, {}".format(pre, source))
