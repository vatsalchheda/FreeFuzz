import paddle
paddle.enable_static()
import paddle.fluid as fluid

dict_dim, emb_dim, hidden_dim = 128, 64, 512
data = fluid.data(name='step_data', shape=[None], dtype='int64')
x = fluid.embedding(input=data, size=[dict_dim, emb_dim])
pre_hidden = fluid.data(
    name='pre_hidden', shape=[None, hidden_dim], dtype='float32')
pre_cell = fluid.data(
    name='pre_cell', shape=[None, hidden_dim], dtype='float32')
hidden = fluid.layers.lstm_unit(
    x_t=x,
    hidden_t_prev=pre_hidden,
    cell_t_prev=pre_cell)