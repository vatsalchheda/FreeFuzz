import paddle.fluid as fluid

dict_dim, emb_dim = 128, 64
data = fluid.data(name='step_data', shape=[None], dtype='int64')
emb = fluid.embedding(input=data, size=[dict_dim, emb_dim])
hidden_dim = 512
x = fluid.layers.fc(input=emb, size=hidden_dim * 3)
pre_hidden = fluid.data(
    name='pre_hidden', shape=[None, hidden_dim], dtype='float32')
hidden = fluid.layers.gru_unit(
    input=x, hidden=pre_hidden, size=hidden_dim * 3)