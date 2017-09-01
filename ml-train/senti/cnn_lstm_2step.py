import pyprind
import jieba
import numpy as np
import re
import tflearn
import time
from sklearn.feature_extraction import FeatureHasher
from tflearn import embedding, bidirectional_rnn, BasicLSTMCell
from tflearn.data_utils import to_categorical
import jieba.posseg as pseg
import tensorflow as tf

from colorlog import ColoredFormatter
import logging


class Logger:
    """Basic logging configuration."""

    def __init__(self, logger_name):
        """init."""
        formatter = ColoredFormatter(
            '%(log_color)s%(levelname)-5s%(reset)s '
            '%(yellow)s[%(asctime)s]%(reset)s%(white)s '
            '%(name)s %(funcName)s %(bold_purple)s:%(lineno)d%(reset)s '
            '%(log_color)s%(message)s%(reset)s',
            datefmt='%y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'blue',
                'INFO': 'bold_cyan',
                'WARNING': 'red,',
                'ERROR': 'bg_bold_red',
                'CRITICAL': 'red,bg_white',
            }
        )

        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)

        # StreamHandler
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(formatter)

        # Add handlers
        logger.addHandler(sh)

        self.logger = logger

    def __getattr__(self, name):  # noqa
        return getattr(self.logger, name)

logger = Logger(__file__).logger

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

    SENTI_MODEL_PATH = "model/20170823_senti"
    SEQ_MODEL_PATH = "model/20170823_seq"

    DICTIONARY_LENGTH = 2 ** 16

    MAX_SENTENCE_LENGTH = 30
    MAX_DOC_SENTENCES_LENGTH = 50

    SENTI_THREADHOLD = 0.7

    URL_PATTERN = r'https://[a-zA-Z0-9.?/&=:]*'
    MARK_PATTERN = '[\s！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'

    SENTI_MODEL = None
    SEQ_MODEL = None

    def __init__(self):
        self.timer = TimeCounter()
        self.comma_tokenizer = lambda x: pseg.cut(x, HMM=True)
        self.vect = FeatureHasher(n_features=self.DICTIONARY_LENGTH, non_negative=True)
        self.n_layer = 3
        self.n_epoch = 20

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

        if type == "senti":
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
        if type == "seq":
            count = 0
            model = self.SENTI_MODEL
            sentences = np.zeros((self.MAX_DOC_SENTENCES_LENGTH, 2), dtype=np.float32)
            for seq in self.preprocess(doc, "senti"):
                predict_result = model.predict([seq])[0]
                sentences.itemset((count, 0), predict_result[0])
                sentences.itemset((count, 1), predict_result[1])
                count += 1
                if count == self.MAX_DOC_SENTENCES_LENGTH:
                    break
            return sentences

    def stream_docs(self, path, label, type):
        x = []
        y = []
        if type == "senti":
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

        elif type == "seq":
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
        doc_streams = [self.stream_docs(path='data/n/train_n3.txt', label=0, type=type),
                       self.stream_docs(path='data/p/train_p3.txt', label=1, type=type)]

        x = []
        y = []

        logger.info("整合訓練集")
        for doc, label in doc_streams:
            x.extend(doc)
            y.extend(label)
        logger.debug("{}, {}".format(np.array(x).shape, np.array(y).shape))
        y = to_categorical(y, nb_classes=2)

        return x, y

    def building_senti_network(self):
        logger.info("建立網路")

        net = tflearn.input_data(shape=[None, self.MAX_SENTENCE_LENGTH], name='input')
        net = embedding(net, input_dim=self.DICTIONARY_LENGTH, output_dim=1024)
        branch1 = tflearn.conv_1d(net, 128, (2, 1024), padding='valid', activation='relu', regularizer="L2")
        branch2 = tflearn.conv_1d(net, 128, (3, 1024), padding='valid', activation='relu', regularizer="L2")
        #branch3 = tflearn.conv_1d(net, 128, (4, 512), padding='valid', activation='relu', regularizer="L2")
        #branch4 = tflearn.avg_pool_1d(net, kernel_size=(4, 2), strides=1)
        #branch4 = tflearn.conv_1d(branch4, 128, (1, 2), padding='valid', activation='relu', regularizer="L2")
        net = tflearn.merge([branch1, branch2], mode='concat', axis=1)

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

    def building_seq_network(self):
        logger.info("建立網路")

        net = tflearn.input_data(shape=[None, self.MAX_DOC_SENTENCES_LENGTH, 2], name='input')
        branch1 = tflearn.conv_1d(net, 128, (1, 2), padding='valid', activation='relu', regularizer="L2")
        branch1 = tflearn.max_pool_1d(branch1, 2)
        branch2 = tflearn.conv_1d(net, 128, (2, 2), padding='valid', activation='relu', regularizer="L2")
        branch2 = tflearn.max_pool_1d(branch2, 2)
        #branch3 = tflearn.conv_1d(net, 128, (4, 2), padding='valid', activation='relu', regularizer="L2")
        #branch4 = tflearn.avg_pool_1d(net, kernel_size=(4, 2), strides=1)
        #branch4 = tflearn.conv_1d(branch4, 128, (1, 2), padding='valid', activation='relu', regularizer="L2")
        net = tflearn.merge([branch1, branch2], mode='concat', axis=1)

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

    def get_senti_model(self):
        model = tflearn.DNN(self.building_senti_network(), tensorboard_verbose=0, tensorboard_dir="log/")
        return model

    def get_seq_model(self):
        model = tflearn.DNN(self.building_seq_network(), tensorboard_verbose=0, tensorboard_dir="log/")
        return model

    def train_senti(self, model_save_path):
        logger.info("訓練senti model")

        model = self.get_senti_model()

        x_train, y_train = self.gen_data("senti")
        early_stopping_cb = EarlyStoppingCallback(val_acc_thresh=0.80)
        try:
            model.fit(x_train, y_train, validation_set=0.1, show_metric=True,
                      batch_size=128, run_id="cnn_lstm_senti", n_epoch=self.n_epoch)
        except Exception as e:
            logger.info("early stop")
            print("{}".format(e))

        model.save(model_save_path)

        return model

    def train_seq(self, model_save_path):
        logger.info("訓練seq model")

        x_train, y_train = self.gen_data("seq")
        early_stopping_cb = EarlyStoppingCallback(val_acc_thresh=0.80)

        try:
            tf.reset_default_graph()
            model = self.get_seq_model()
            model.fit(x_train, y_train, validation_set=0.1, show_metric=True,
                      batch_size=10, run_id="cnn_lstm_seq", n_epoch=self.n_epoch)
            model.save(model_save_path)
        except Exception as e:
            logger.info("early stop")
            print("{}".format(e))

        return model

    def go_deep(self):
        if self.TRAINING:
            # SENTI
            senti_model = self.train_senti(self.SENTI_MODEL_PATH)
            self.SENTI_MODEL = senti_model
            # SENTI
            #self.SENTI_MODEL = self.get_senti_model()
            #self.SENTI_MODEL.load(self.SENTI_MODEL_PATH)

            # SEQ
            seq_model = self.train_seq(self.SEQ_MODEL_PATH)
            self.SEQ_MODEL = seq_model

        else:
            # SENTI
            self.SENTI_MODEL = self.get_senti_model()
            self.SENTI_MODEL.load(self.SENTI_MODEL_PATH)

            tf.reset_default_graph()

            # SEQ
            self.SEQ_MODEL = self.get_seq_model()
            self.SEQ_MODEL.load(self.SEQ_MODEL_PATH)

        return self

    def predict_senti(self, doc):
        x = self.preprocess(doc, type="senti")
        #print("senti preprocess : {}".format(x))
        return self.SENTI_MODEL.predict(x)

    def predict(self, doc):
        x = self.preprocess(doc, type="seq")
        #print("seq preprocess : {}".format(x))
        return self.SEQ_MODEL.predict([x])


ob = SentimentClassifier()
model = ob.go_deep()

while True:
    input_str = input("說說話吧: ")
    result = model.predict_senti(input_str)
    print(result)

    result = model.predict(input_str)
    print("######\n[{:05f},{:05f}]\n######".format(result[0][0], result[0][1]))

