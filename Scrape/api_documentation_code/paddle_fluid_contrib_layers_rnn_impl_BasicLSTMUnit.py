import paddle.fluid.layers as layers
from paddle.fluid.contrib.layers import BasicLSTMUnit

input_size = 128
hidden_size = 256
input = layers.data( name = "input", shape = [-1, input_size], dtype='float32')
pre_hidden = layers.data( name = "pre_hidden", shape=[-1, hidden_size], dtype='float32')
pre_cell = layers.data( name = "pre_cell", shape=[-1, hidden_size], dtype='float32')

lstm_unit = BasicLSTMUnit( "gru_unit", hidden_size)

new_hidden, new_cell = lstm_unit( input, pre_hidden, pre_cell )