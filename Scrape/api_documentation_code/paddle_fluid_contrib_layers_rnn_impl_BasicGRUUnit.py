import paddle.fluid.layers as layers
from paddle.fluid.contrib.layers import BasicGRUUnit

input_size = 128
hidden_size = 256
input = layers.data( name = "input", shape = [-1, input_size], dtype='float32')
pre_hidden = layers.data( name = "pre_hidden", shape=[-1, hidden_size], dtype='float32')

gru_unit = BasicGRUUnit( "gru_unit", hidden_size )

new_hidden = gru_unit( input, pre_hidden )