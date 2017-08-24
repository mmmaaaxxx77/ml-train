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

    TF_MODEL_PATH = "model/20170823_cnn_lstm_tf"
    SGD_MODEL_PATH = "model/20170823_cnn_lstm_sgd"

    DICTIONARY_LENGTH = 2 ** 15

    MAX_SENTENCE_LENGTH = 10
    MAX_DOC_SENTENCES_LENGTH = 20

    SENTI_THREADHOLD = 0.7

    URL_PATTERN = r'https://[a-zA-Z0-9.?/&=:]*'
    MARK_PATTERN = '[\s！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'

    TF_MODEL = None
    SGD_MODEL = None

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

    def preprocess(self, doc, type):
        # 去換行
        doc = self.clear_new_line(doc)
        # 斷詞
        words = self.comma_tokenizer(doc)

        if type == "tf":
            sentences = []
            sentence = np.zeros((self.MAX_SENTENCE_LENGTH, ), dtype=np.int32)
            sentence_count = 0
            for word, flag in words:
                # 非標點符號
                if flag != "x":
                    if sentence_count < self.MAX_SENTENCE_LENGTH:
                        sentence.itemset((sentence_count,), self.transform_word_to_id(word))
                        sentence_count += 1
                else:
                    nonzero = np.count_nonzero(sentence)
                    if nonzero > 0:
                        sentence_count = 0
                        sentences.append(sentence)
                    sentence = np.zeros((self.MAX_SENTENCE_LENGTH,), dtype=np.int32)
                    continue
            return sentences
        if type == "sgd":
            sentences = [3] * self.MAX_DOC_SENTENCES_LENGTH
            count = 0
            model = self.TF_MODEL
            for seq in self.preprocess(doc, "tf"):
                predict_result = model.predict([seq])[0]
                if count == self.MAX_DOC_SENTENCES_LENGTH:
                    break
                label = 3
                if predict_result[0] < predict_result[1] and self.SENTI_THREADHOLD <= predict_result[1]:
                    label = 1
                elif predict_result[0] > predict_result[1] and self.SENTI_THREADHOLD <= predict_result[0]:
                    label = 0
                sentences[count] = label
                count += 1
            return sentences

    def stream_docs(self, path, label, type):
        x = []
        y = []
        if type == "tf":
            lines_count = 0
            with open(path, 'r') as file:
                lines_count = sum(1 for _ in file)

            logger.info("處理檔案 PATH: {}, COUNT: {}".format(path, lines_count))

            with open(path, 'r') as file:
                pbar = pyprind.ProgBar(lines_count)
                for doc in file:
                    pbar.update()

                    for sents in self.preprocess(doc, type):
                        x.append(sents)
                        y.append(label)

        elif type == "sgd":
            lines_count = 0
            with open(path, 'r') as file:
                lines_count = sum(1 for _ in file)

            logger.info("處理檔案 PATH: {}, COUNT: {}".format(path, lines_count))

            with open(path, 'r') as file:
                pbar = pyprind.ProgBar(lines_count)
                for doc in file:
                    pbar.update()
                    x.append(self.preprocess(doc, type))
                    y.append(label)

        return x, y

    def gen_data(self, type):
        doc_streams = [self.stream_docs(path='data/n/train_n2.txt', label=0, type=type),
                       self.stream_docs(path='data/p/train_p2.txt', label=1, type=type)]

        x = []
        y = []

        logger.info("整合訓練集")
        if type == "tf":
            for doc, label in doc_streams:
                x.extend(doc)
                y.extend(label)

            logger.debug("{}, {}".format(np.array(x).shape, np.array(y).shape))
            y = to_categorical(y, nb_classes=2)
        elif type == "sgd":
            for doc, label in doc_streams:
                x.extend(doc)
                y.extend(label)

            logger.debug("{}, {}".format(np.array(x).shape, np.array(y).shape))

        return x, y

    def building_network(self):
        logger.info("建立網路")

        net = tflearn.input_data(shape=[None, self.MAX_SENTENCE_LENGTH], name='input')
        net = embedding(net, input_dim=self.DICTIONARY_LENGTH, output_dim=512)
        """
        branch1 = tflearn.conv_1d(net, 128, 1, activation='relu', regularizer="L2")
        branch2 = tflearn.conv_1d(net, 128, 2, activation='relu', regularizer="L2")
        branch3 = tflearn.conv_1d(net, 128, 3, activation='relu', regularizer="L2")
        net = tflearn.merge([branch1, branch2, branch3], mode='concat', axis=0)
        net = tflearn.dropout(net, 0.5)
        """

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

    def train_tf(self, model_save_path):
        logger.info("訓練tf model")

        model = self.get_model()

        x_train, y_train = self.gen_data("tf")
        early_stopping_cb = EarlyStoppingCallback(val_acc_thresh=0.85)
        try:
            model.fit(x_train, y_train, validation_set=0.1, show_metric=True,
                      batch_size=128, run_id="cnn_lstm", n_epoch=5, callbacks=early_stopping_cb)
        except Exception as e:
            logger.info("early stop")
            print("{}".format(e))

        model.save(model_save_path)

        return model

    def train_sgd(self, model_save_path):
        logger.info("訓練SGD model")

        x_train, y_train = self.gen_data("sgd")
        clf = SGDClassifier(loss="log", verbose=1)
        clf.fit(x_train, y_train)
        joblib.dump(clf, model_save_path)

        logger.info("SGD VAL ACC: {}".format(clf.score(x_train, y_train)))

        return clf

    def go_deep(self):
        if self.TRAINING:
            # TF
            tf_model = self.train_tf(self.TF_MODEL_PATH)
            self.TF_MODEL = tf_model

            # SGD
            sgd_model = self.train_sgd(self.SGD_MODEL_PATH)
            self.SGD_MODEL = sgd_model

        else:
            # TF
            self.TF_MODEL = self.get_model()
            self.TF_MODEL.load(self.TF_MODEL_PATH)

            # SGD
            sgd_model = joblib.load(self.SGD_MODEL_PATH)
            self.SGD_MODEL = sgd_model

        return self

    def predict_tf(self, doc):
        x = self.preprocess(doc, type="tf")
        print("tf preprocess : {}".format(x))
        return self.TF_MODEL.predict(x)

    def predict(self, doc):
        x = self.preprocess(doc, type="sgd")
        print("sgd preprocess : {}".format(x))
        return self.SGD_MODEL.predict([x])[0], self.SGD_MODEL.predict_proba([x])[0]


ob = SentimentClassifier()
model = ob.go_deep()

while True:
    input_str = input("說說話吧: ")
    result = model.predict_tf(input_str)
    print(result)

    result = model.predict(input_str)
    print(result)

