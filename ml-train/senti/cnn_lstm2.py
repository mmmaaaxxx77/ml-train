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
import jieba.posseg as pseg
import tensorflow as tf
from senti import logconf

logger = logconf.Logger(__file__).logger

jieba.load_userdict("dict/dict.txt.big.txt")
jieba.load_userdict("dict/NameDict_Ch_v2")
jieba.load_userdict("dict/ntusd-negative.txt")
jieba.load_userdict("dict/ntusd-positive.txt")
jieba.load_userdict("dict/鄉民擴充辭典.txt")

print("###### 測試斷詞 ######")
words = pseg.cut('柯P,是柯市長，真不錯＾Ｐ＾', HMM=False)
for word, flag in words:
    print("{} | {}".format(word, flag))
print("###### 測試斷詞 END ######")


class TimeCounter:
    def __init__(self):
        self.tStart = time.time()

    def print_time(self):
        tEnd = time.time()
        print("It cost {:0.2f} sec".format(tEnd - self.tStart))


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


class SentimentClassifier:

    TRAINING = True

    TF_MODEL_PATH = "model/20170823_cnn_lstm"

    DICTIONARY_LENGTH = 2 ** 15

    MAX_SENTENCE_LENGTH = 10
    MAX_DOC_SENTENCES_LENGTH = 20

    URL_PATTERN = r'https://[a-zA-Z0-9.?/&=:]*'
    MARK_PATTERN = '[\s！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'

    def __init__(self):
        self.timer = TimeCounter()
        self.comma_tokenizer = lambda x: pseg.cut(x, HMM=True)
        self.vect = FeatureHasher(n_features=self.DICTIONARY_LENGTH, non_negative=True)
        self.n_layer = 1

    def clear_doc(self, doc):
        # 去除網址和符號
        text = re.sub(self.URL_PATTERN, '', doc)
        text = re.sub(self.MARK_PATTERN, '', text)
        return text

    def clear_new_line(self, doc):
        text = re.sub('\n', '', doc)
        return text

    def transform_word_to_id(self, word):
        terms = [{word: 1}]
        ids_sci = self.vect.transform(terms)
        return ids_sci[0].nonzero()[1][0]

    def preprocess(self, doc):
        # 去換行
        doc = self.clear_new_line(doc)
        # 斷詞
        words = self.comma_tokenizer(doc)

        sentences = np.zeros((self.MAX_DOC_SENTENCES_LENGTH, self.MAX_SENTENCE_LENGTH), dtype=np.int32)
        sentence_count = 0
        sentences_count = 0
        for word, flag in words:
            # 非標點符號
            if flag != "x":
                if sentence_count < self.MAX_SENTENCE_LENGTH:
                    #sentences.itemset((sentences_count, sentence_count,  self.transform_word_to_id(word)), 1)
                    sentences.itemset((sentences_count, sentence_count), self.transform_word_to_id(word))
                    # sentences[sentences_count][sentence_count] = self.transform_word_to_id(word)
                    sentence_count += 1
            else:
                nonzero = np.count_nonzero(sentences[sentences_count])
                if nonzero > 0:
                    sentences_count += 1
                    sentence_count = 0

                    if sentences_count == self.MAX_DOC_SENTENCES_LENGTH:
                        break
                continue
        return sentences

    def stream_docs(self, path, label):
        x = []
        y = []

        lines_count = 0
        with open(path, 'r') as file:
            lines_count = sum(1 for _ in file)

        logger.info("處理檔案 PATH: {}, COUNT: {}".format(path, lines_count))

        with open(path, 'r') as file:
            pbar = pyprind.ProgBar(lines_count)
            for doc in file:
                pbar.update()

                x.append(self.preprocess(doc))
                y.append(label)
        return x, y

    def gen_data(self):
        doc_streams = [self.stream_docs(path='data/n/train_n2.txt', label=0),
                       self.stream_docs(path='data/p/train_p2.txt', label=1)]

        x = []
        y = []

        logger.info("整合訓練集")
        for doc, label in doc_streams:
            x.extend(doc)
            y.extend(label)

        logger.debug("{}, {}".format(np.array(x).shape, np.array(y).shape))
        y = to_categorical(y, nb_classes=2)
        return x, y

    def building_network(self):
        logger.info("建立網路")

        net = tflearn.input_data(shape=[None, self.MAX_DOC_SENTENCES_LENGTH, self.MAX_SENTENCE_LENGTH], name='input')
        branch1 = tflearn.conv_1d(net, 128, 1, activation='relu', regularizer="L2")
        branch2 = tflearn.conv_1d(net, 128, 2, activation='relu', regularizer="L2")
        branch3 = tflearn.conv_1d(net, 128, 3, activation='relu', regularizer="L2")
        net = tflearn.merge([branch1, branch2, branch3], mode='concat', axis=1)
        net = tflearn.dropout(net, 0.5)

        for n in range(1, self.n_layer):
            net = bidirectional_rnn(
                net,
                BasicLSTMCell(128),
                BasicLSTMCell(128),
                return_seq=True,
            )
            net = tflearn.dropout(net, 0.8)
        net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128))
        net = tflearn.dropout(net, 0.8)
        net = tflearn.fully_connected(net, 2, activation='softmax')
        net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                                 loss='categorical_crossentropy')
        return net

    def get_model(self):
        model = tflearn.DNN(self.building_network(), tensorboard_verbose=0, tensorboard_dir="log/")
        return model

    def train(self, model_save_path):
        logger.info("訓練model")

        model = self.get_model()

        x_train, y_train = self.gen_data()
        early_stopping_cb = EarlyStoppingCallback(val_acc_thresh=0.85)
        model.fit(x_train, y_train, validation_set=0.1, show_metric=True,
                  batch_size=10, run_id="cnn_lstm", n_epoch=100, callbacks=early_stopping_cb)
        model.save(model_save_path)

        return model

    def go_deep(self):
        if self.TRAINING:
            model = self.train(self.TF_MODEL_PATH)
        else:
            model = self.get_model().load(self.TF_MODEL_PATH)

        return model

ob = SentimentClassifier()
model = ob.go_deep()
while True:
    input_str = input("說說話吧: ")
    x = ob.preprocess(input_str)
    print(x.tolist())
    result = model.predict([x])
    print(result[0])
