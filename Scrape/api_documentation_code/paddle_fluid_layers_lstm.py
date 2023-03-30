import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
paddle.enable_static()

emb_dim = 256
vocab_size = 10000
data = fluid.data(name='x', shape=[None, 100], dtype='int64')
emb = fluid.embedding(input=data, size=[vocab_size, emb_dim], is_sparse=True)
batch_size = 100
dropout_prob = 0.2
input_size = 100
hidden_size = 150
num_layers = 1
max_len = 12
init_h = layers.fill_constant( [num_layers, batch_size, hidden_size], 'float32', 0.0 )
init_c = layers.fill_constant( [num_layers, batch_size, hidden_size], 'float32', 0.0 )
rnn_out, last_h, last_c = layers.lstm( emb, init_h, init_c, \
        max_len, hidden_size, num_layers, \
        dropout_prob=dropout_prob)
rnn_out.shape  # (-1, 100, 150)
last_h.shape  # (1, 20, 150)
last_c.shape  # (1, 20, 150)