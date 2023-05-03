import paddle
paddle.enable_static()
import paddle.fluid.layers as layers
from paddle.fluid.contrib.layers import basic_lstm

batch_size = 20
input_size = 128
hidden_size = 256
num_layers = 2
dropout = 0.5
bidirectional = True
batch_first = False

input = layers.data( name = "input", shape = [-1, batch_size, input_size], dtype='float32')
pre_hidden = layers.data( name = "pre_hidden", shape=[-1, hidden_size], dtype='float32')
pre_cell = layers.data( name = "pre_cell", shape=[-1, hidden_size], dtype='float32')
sequence_length = layers.data( name="sequence_length", shape=[-1], dtype='int32')

rnn_out, last_hidden, last_cell = basic_lstm( input, pre_hidden, pre_cell, \
        hidden_size, num_layers = num_layers, \
        sequence_length = sequence_length, dropout_prob=dropout, bidirectional = bidirectional, \
        batch_first = batch_first)