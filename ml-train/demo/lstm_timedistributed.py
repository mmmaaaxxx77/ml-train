import numpy as np
import tflearn
from sklearn.preprocessing import OneHotEncoder
from tflearn import embedding, bidirectional_rnn, BasicLSTMCell
from tflearn.data_utils import to_categorical, pad_sequences

x = np.array([
    [1, 2, 3],
    [1, 3, 4],
    [1, 4, 5],
    [1, 2, 3],
    [1, 3, 4],
    [1, 4, 5],
    [2, 5, 3],
    [2, 4, 3],
    [2, 5, 2],
    [2, 5, 3],
    [2, 4, 3],
    [2, 5, 2],
    [3, 5, 3],
    [3, 4, 3],
    [3, 4, 5],
    [3, 5, 3],
    [3, 4, 3],
    [3, 4, 5],
])

y = np.array([
    1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3
])
enc = OneHotEncoder()
enc.fit(y)
print(enc.transform(y).toarray())

seq = np.array([i/float(5) for i in range(5)])
print(seq)
print(seq.reshape(1, 5, 1))
y = to_categorical(y, nb_classes=4)

test_x = [
    [2, 2, 2],
    [3, 3, 3]
]
test_y = [2, 3]
test_y = to_categorical(test_y, nb_classes=4)

net = tflearn.input_data(shape=[None, 3])
net = embedding(net, input_dim=6, output_dim=5)
net = tflearn.lstm(net, 128, return_seq=True)
net = tflearn.dropout(net, 0.8)
net = tflearn.time_distributed(net, tflearn.fully_connected, [1])
net = tflearn.softmax(net)
net = tflearn.reshape(net, [-1, 2, 4])
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')
model = tflearn.DNN(net, tensorboard_verbose=3, tensorboard_dir="log/")
model.fit(x, y, validation_set=(test_x, test_y), show_metric=True,
          batch_size=2, run_id="demo_lstm_test_timedistributed_dr0.8", n_epoch=50)
