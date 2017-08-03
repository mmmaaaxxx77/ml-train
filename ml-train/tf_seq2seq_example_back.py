from __future__ import division, print_function

import os
import tflearn
import json

import numpy as np
import tensorflow as tf


# -----------------------------------------------------------------------------
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.contrib.rnn import MultiRNNCell, BasicLSTMCell
from tensorflow.contrib.rnn.python.ops import rnn_cell


class SequencePattern(object):
    INPUT_SEQUENCE_LENGTH = 10
    OUTPUT_SEQUENCE_LENGTH = 10
    INPUT_MAX_INT = 20
    OUTPUT_MAX_INT = 20
    PATTERN_NAME = "sorted"

    def __init__(self, name=None, in_seq_len=None, out_seq_len=None):
        if name is not None:
            assert hasattr(self, "%s_sequence" % name)
            self.PATTERN_NAME = name
        if in_seq_len:
            self.INPUT_SEQUENCE_LENGTH = in_seq_len
        if out_seq_len:
            self.OUTPUT_SEQUENCE_LENGTH = out_seq_len

    def generate_output_sequence(self, x):
        '''
        For a given input sequence, generate the output sequence.  x is a 1D numpy array 
        of integers, with length INPUT_SEQUENCE_LENGTH.

        Returns a 1D numpy array of length OUTPUT_SEQUENCE_LENGTH

        This procedure defines the pattern which the seq2seq RNN will be trained to find.
        '''
        return getattr(self, "%s_sequence" % self.PATTERN_NAME)(x)

    def maxmin_dup_sequence(self, x):
        '''
        Generate sequence with [max, min, rest of original entries]
        '''
        x = np.array(x)
        y = [x.max(), x.min()] + list(x[2:])
        return np.array(y)[:self.OUTPUT_SEQUENCE_LENGTH]  # truncate at out seq len

    def sorted_sequence(self, x):
        '''
        Generate sorted version of original sequence
        '''
        return np.array(sorted(x))[:self.OUTPUT_SEQUENCE_LENGTH]

    def reversed_sequence(self, x):
        '''
        Generate reversed version of original sequence
        '''
        return np.array(x[::-1])[:self.OUTPUT_SEQUENCE_LENGTH]


# -----------------------------------------------------------------------------

class TFLearnSeq2Seq(object):
    '''
    seq2seq recurrent neural network, implemented using TFLearn.
    '''
    AVAILABLE_MODELS = ["embedding_rnn", "embedding_attention"]

    def __init__(self, sequence_pattern, seq2seq_model=None, verbose=None, name=None, data_dir=None):
        '''
        sequence_pattern_class = a SequencePattern class instance, which defines pattern parameters 
                                 (input, output lengths, name, generating function)
        seq2seq_model = string specifying which seq2seq model to use, e.g. "embedding_rnn"
        '''
        self.sequence_pattern = sequence_pattern
        self.seq2seq_model = seq2seq_model or "embedding_rnn"
        assert self.seq2seq_model in self.AVAILABLE_MODELS
        self.in_seq_len = self.sequence_pattern.INPUT_SEQUENCE_LENGTH
        self.out_seq_len = self.sequence_pattern.OUTPUT_SEQUENCE_LENGTH
        self.in_max_int = self.sequence_pattern.INPUT_MAX_INT
        self.out_max_int = self.sequence_pattern.OUTPUT_MAX_INT
        self.verbose = verbose or 0
        self.n_input_symbols = self.in_max_int + 1
        self.n_output_symbols = self.out_max_int + 2  # extra one for GO symbol
        self.model_instance = None
        self.name = name
        self.data_dir = data_dir

    def generate_trainig_data(self, num_points):
        '''
        Generate training dataset.  Produce random (integer) sequences X, and corresponding
        expected output sequences Y = generate_output_sequence(X).

        Return xy_data, y_data (both of type uint32)

        xy_data = numpy array of shape [num_points, in_seq_len + out_seq_len], with each point being X + Y
        y_data  = numpy array of shape [num_points, out_seq_len]
        '''
        x_data = np.random.randint(0, self.in_max_int,
                                   size=(num_points, self.in_seq_len))  # shape [num_points, in_seq_len]
        x_data = x_data.astype(np.uint32)  # ensure integer type

        y_data = [self.sequence_pattern.generate_output_sequence(x) for x in x_data]
        y_data = np.array(y_data)

        xy_data = np.append(x_data, y_data, axis=1)  # shape [num_points, 2*seq_len]
        return xy_data, y_data

    def sequence_loss(self, y_pred, y_true):
        '''
        Loss function for the seq2seq RNN.  Reshape predicted and true (label) tensors, generate dummy weights,
        then use seq2seq.sequence_loss to actually compute the loss function.
        '''
        if self.verbose > 2: print("my_sequence_loss y_pred=%s, y_true=%s" % (y_pred, y_true))
        logits = tf.unstack(y_pred, axis=1)  # list of [-1, num_decoder_synbols] elements
        targets = tf.unstack(y_true,
                             axis=1)  # y_true has shape [-1, self.out_seq_len]; unpack to list of self.out_seq_len [-1] elements
        if self.verbose > 2:
            print("my_sequence_loss logits=%s" % (logits,))
            print("my_sequence_loss targets=%s" % (targets,))
        weights = [tf.ones_like(yp, dtype=tf.float32) for yp in targets]
        if self.verbose > 4: print("my_sequence_loss weights=%s" % (weights,))
        sl = seq2seq.sequence_loss(logits, targets, weights)
        if self.verbose > 2: print("my_sequence_loss return = %s" % sl)
        return sl

    def accuracy(self, y_pred, y_true,
                 x_in):  # y_pred is [-1, self.out_seq_len, num_decoder_symbols]; y_true is [-1, self.out_seq_len]
        '''
        Compute accuracy of the prediction, based on the true labels.  Use the average number of equal
        values.
        '''
        pred_idx = tf.to_int32(tf.argmax(y_pred, 2))  # [-1, self.out_seq_len]
        if self.verbose > 2: print("my_accuracy pred_idx = %s" % pred_idx)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_idx, y_true), tf.float32), name='acc')
        return accuracy

    def model(self, mode="train", num_layers=1, cell_size=32, cell_type="BasicLSTMCell", embedding_size=20,
              learning_rate=0.0001,
              tensorboard_verbose=0, checkpoint_path=None):
        '''
        Build tensor specifying graph of operations for the seq2seq neural network model.

        mode = string, either "train" or "predict"
        cell_type = attribute of rnn_cell specifying which RNN cell type to use
        cell_size = size for the hidden layer in the RNN cell
        num_layers = number of RNN cell layers to use

        Return TFLearn model instance.  Use DNN model for this.
        '''
        assert mode in ["train", "predict"]

        checkpoint_path = checkpoint_path or (
        "%s%ss2s_checkpoint.tfl" % (self.data_dir or "", "/" if self.data_dir else ""))
        GO_VALUE = self.out_max_int + 1  # unique integer value used to trigger decoder outputs in the seq2seq RNN

        network = tflearn.input_data(shape=[None, self.in_seq_len + self.out_seq_len], dtype=tf.int32, name="XY")
        encoder_inputs = tf.slice(network, [0, 0], [-1, self.in_seq_len], name="enc_in")  # get encoder inputs
        encoder_inputs = tf.unstack(encoder_inputs,
                                    axis=1)  # transform into list of self.in_seq_len elements, each [-1]

        decoder_inputs = tf.slice(network, [0, self.in_seq_len], [-1, self.out_seq_len],
                                  name="dec_in")  # get decoder inputs
        decoder_inputs = tf.unstack(decoder_inputs,
                                    axis=1)  # transform into list of self.out_seq_len elements, each [-1]

        go_input = tf.multiply(tf.ones_like(decoder_inputs[0], dtype=tf.int32),
                               GO_VALUE)  # insert "GO" symbol as the first decoder input; drop the last decoder input
        decoder_inputs = [go_input] + decoder_inputs[
                                      : self.out_seq_len - 1]  # insert GO as first; drop last decoder input

        feed_previous = not (mode == "train")

        self.n_input_symbols = self.in_max_int + 1  # default is integers from 0 to 9
        self.n_output_symbols = self.out_max_int + 2  # extra "GO" symbol for decoder inputs

        single_cell = BasicLSTMCell(cell_size, state_is_tuple=True)
        if num_layers == 1:
            cell = single_cell
        else:
            cell = MultiRNNCell([single_cell] * num_layers)

        if self.seq2seq_model == "embedding_rnn":
            model_outputs, states = seq2seq.embedding_rnn_seq2seq(encoder_inputs,
                                                                  # encoder_inputs: A list of 2D Tensors [batch_size, input_size].
                                                                  decoder_inputs,
                                                                  cell,
                                                                  num_encoder_symbols=self.n_input_symbols,
                                                                  num_decoder_symbols=self.n_output_symbols,
                                                                  embedding_size=embedding_size,
                                                                  feed_previous=feed_previous)
        elif self.seq2seq_model == "embedding_attention":
            model_outputs, states = seq2seq.embedding_attention_seq2seq(encoder_inputs,
                                                                        # encoder_inputs: A list of 2D Tensors [batch_size, input_size].
                                                                        decoder_inputs,
                                                                        cell,
                                                                        num_encoder_symbols=self.n_input_symbols,
                                                                        num_decoder_symbols=self.n_output_symbols,
                                                                        embedding_size=embedding_size,
                                                                        num_heads=1,
                                                                        initial_state_attention=False,
                                                                        feed_previous=feed_previous)
        else:
            raise Exception('[TFLearnSeq2Seq] Unknown seq2seq model %s' % self.seq2seq_model)

        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + "seq2seq_model",
                             model_outputs)  # for TFLearn to know what to save and restore

        network = tf.stack(model_outputs, axis=1)

        with tf.name_scope("TargetsData"):  # placeholder for target variable (i.e. trainY input)
            targetY = tf.placeholder(shape=[None, self.out_seq_len], dtype=tf.int32, name="Y")

        network = tflearn.regression(network,
                                     placeholder=targetY,
                                     optimizer='adam',
                                     learning_rate=learning_rate,
                                     loss=self.sequence_loss,
                                     metric=self.accuracy,
                                     name="Y")

        model = tflearn.DNN(network, tensorboard_verbose=tensorboard_verbose, checkpoint_path=checkpoint_path)
        return model

    def train(self, num_epochs=20, num_points=100000, model=None, model_params=None, weights_input_fn=None,
              validation_set=0.1, snapshot_step=5000, batch_size=128, weights_output_fn=None):
        '''
        Train model, with specified number of epochs, and dataset size.

        Use specified model, or create one if not provided.  Load initial weights from file weights_input_fn, 
        if provided. validation_set specifies what to use for the validation.

        Returns logits for prediction, as an numpy array of shape [out_seq_len, n_output_symbols].
        '''
        trainXY, trainY = self.generate_trainig_data(num_points)
        print("[TFLearnSeq2Seq] Training on %d point dataset (pattern '%s'), with %d epochs" % (num_points,
                                                                                                self.sequence_pattern.PATTERN_NAME,
                                                                                                num_epochs))
        if self.verbose > 1:
            print("  model parameters: %s" % json.dumps(model_params, indent=4))
        model_params = model_params or {}
        model = model or self.setup_model("train", model_params, weights_input_fn)

        model.fit(trainXY, trainY,
                  n_epoch=num_epochs,
                  validation_set=validation_set,
                  batch_size=batch_size,
                  #shuffle=True,
                  #show_metric=True,
                  #snapshot_step=snapshot_step,
                  #snapshot_epoch=False,
                  run_id="TFLearnSeq2Seq"
                  )
        print("Done!")
        if weights_output_fn is not None:
            weights_output_fn = self.canonical_weights_fn(weights_output_fn)
            model.save(weights_output_fn)
            print("Saved %s" % weights_output_fn)
            self.weights_output_fn = weights_output_fn
        return model

    def canonical_weights_fn(self, iteration_num=0):
        '''
        Construct canonical weights filename, based on model and pattern names.
        '''
        if not type(iteration_num) == int:
            try:
                iteration_num = int(iteration_num)
            except Exception as err:
                return iteration_num
        model_name = self.name or "basic"
        wfn = "ts2s__%s__%s_%s.tfl" % (model_name, self.sequence_pattern.PATTERN_NAME, iteration_num)
        if self.data_dir:
            wfn = "%s/%s" % (self.data_dir, wfn)
        self.weights_filename = wfn
        return wfn

    def setup_model(self, mode, model_params=None, weights_input_fn=None):
        '''
        Setup a model instance, using the specified mode and model parameters.
        Load the weights from the specified file, if it exists.
        If weights_input_fn is an integer, use that the model name, and
        the pattern name, to construct a canonical filename.
        '''
        model_params = model_params or {}
        model = self.model_instance or self.model(mode=mode, **model_params)
        self.model_instance = model
        if weights_input_fn:
            if type(weights_input_fn) == int:
                weights_input_fn = self.canonical_weights_fn(weights_input_fn)
            if os.path.exists(weights_input_fn):
                model.load(weights_input_fn)
                print("[TFLearnSeq2Seq] model weights loaded from %s" % weights_input_fn)
            else:
                print("[TFLearnSeq2Seq] MISSING model weights file %s" % weights_input_fn)
        return model

    def predict(self, Xin, model=None, model_params=None, weights_input_fn=None):
        '''
        Make a prediction, using the seq2seq model, for the given input sequence Xin.
        If model is not provided, create one (or use last created instance).

        Return prediction, y

        prediction = array of integers, giving output prediction.  Length = out_seq_len
        y = array of shape [out_seq_len, out_max_int], giving logits for output prediction
        '''
        if not model:
            model = self.model_instance or self.setup_model("predict", model_params, weights_input_fn)

        if self.verbose: print("Xin = %s" % str(Xin))

        X = np.array(Xin).astype(np.uint32)
        assert len(X) == self.in_seq_len
        if self.verbose:
            print("X Input shape=%s, data=%s" % (X.shape, X))
            print("Expected output = %s" % str(self.sequence_pattern.generate_output_sequence(X)))

        Yin = [0] * self.out_seq_len

        XY = np.append(X, np.array(Yin).astype(np.float32))
        XY = XY.reshape([-1, self.in_seq_len + self.out_seq_len])  # batch size 1
        if self.verbose > 1: print("XY Input shape=%s, data=%s" % (XY.shape, XY))

        res = model.predict(XY)
        res = np.array(res)
        if self.verbose > 1: print("prediction shape = %s" % str(res.shape))
        y = res.reshape(self.out_seq_len, self.n_output_symbols)
        prediction = np.argmax(y, axis=1)
        if self.verbose:
            print("Predicted output sequence: %s" % str(prediction))
        return prediction, y

# -----------------------------------------------------------------------------
# unit tests


def test_sp1():
    '''
    Test two different SequencePattern instances
    '''
    sp = SequencePattern("maxmin_dup")
    y = sp.generate_output_sequence(range(10))
    assert all(y == np.array([9, 0, 2, 3, 4, 5, 6, 7, 8, 9]))
    print(y)
    sp = SequencePattern("sorted")
    y = sp.generate_output_sequence([5, 6, 1, 2, 9])
    assert all(y == np.array([1, 2, 5, 6, 9]))
    print(y)
    sp = SequencePattern("reversed")
    y = sp.generate_output_sequence(range(10))
    assert all(y == np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]))
    print(y)


def test_sp2():
    '''
    Test two SequencePattern instance with lengths different from default
    '''
    sp = SequencePattern("sorted", in_seq_len=20, out_seq_len=5)
    x = np.random.randint(0, 9, 20)
    y = sp.generate_output_sequence(x)
    assert len(y) == 5
    print(y)
    y_exp = sorted(x)[:5]
    assert all(y == y_exp)
    print(y)


def test_train1():
    '''
    Test simple training of an embedding_rnn seq2seq model
    '''
    sp = SequencePattern()
    ts2s = TFLearnSeq2Seq(sp, verbose=2)
    ofn = "test_%s" % ts2s.canonical_weights_fn(0)
    print("using weights filename %s" % ofn)
    if os.path.exists(ofn):
        os.unlink(ofn)
    tf.reset_default_graph()
    ts2s.train(num_epochs=1, num_points=10000, weights_output_fn=ofn)


def test_predict1():
    '''
    Test simple preductions using weights just produced (in test_train1)
    '''
    sp = SequencePattern()
    ts2s = TFLearnSeq2Seq(sp, verbose=2)
    wfn = "test_%s" % ts2s.canonical_weights_fn(0)
    print("using weights filename %s" % wfn)
    tf.reset_default_graph()
    prediction, y = ts2s.predict(Xin=range(10), weights_input_fn=wfn)
    print("{},{}".format(prediction, y))
    assert len(prediction == 10)


def test_train_predict2():
    '''
    Test that the embedding_attention model works, with saving and loading of weights
    '''
    import tempfile
    sp = SequencePattern()
    tempdir = tempfile.mkdtemp()
    ts2s = TFLearnSeq2Seq(sp, seq2seq_model="embedding_attention", data_dir=tempdir, name="attention")
    tf.reset_default_graph()
    ts2s.train(num_epochs=1, num_points=1000, weights_output_fn=1, weights_input_fn=0)
    assert os.path.exists(ts2s.weights_output_fn)

    tf.reset_default_graph()
    ts2s = TFLearnSeq2Seq(sp, seq2seq_model="embedding_attention", data_dir="DATA", name="attention", verbose=1)
    prediction, y = ts2s.predict(Xin=range(10), weights_input_fn=1)
    assert len(prediction == 10)

    os.system("rm -rf %s" % tempdir)


def test_train_predict3():
    '''
    Test that a model trained on sequencees of one length can be used for predictions on other sequence lengths
    '''
    import tempfile
    sp = SequencePattern("sorted", in_seq_len=10, out_seq_len=10)
    tempdir = tempfile.mkdtemp()
    ts2s = TFLearnSeq2Seq(sp, seq2seq_model="embedding_attention", data_dir=tempdir, name="attention")
    tf.reset_default_graph()
    ts2s.train(num_epochs=1, num_points=1000, weights_output_fn=1, weights_input_fn=0)
    assert os.path.exists(ts2s.weights_output_fn)

    tf.reset_default_graph()
    sp = SequencePattern("sorted", in_seq_len=20, out_seq_len=8)
    tf.reset_default_graph()
    ts2s = TFLearnSeq2Seq(sp, seq2seq_model="embedding_attention", data_dir="DATA", name="attention", verbose=1)
    x = np.random.randint(0, 9, 20)
    prediction, y = ts2s.predict(x, weights_input_fn=1)
    assert len(prediction == 8)

    os.system("rm -rf %s" % tempdir)

#test_train1()
test_predict1()
