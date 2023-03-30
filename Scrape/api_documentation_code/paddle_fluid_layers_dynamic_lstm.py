import paddle.fluid as fluid
emb_dim = 256
vocab_size = 10000
hidden_dim = 512

data = fluid.data(name='x', shape=[None], dtype='int64', lod_level=1)
emb = fluid.embedding(input=data, size=[vocab_size, emb_dim], is_sparse=True)

forward_proj = fluid.layers.fc(input=emb, size=hidden_dim * 4,
                               bias_attr=False)

forward, cell = fluid.layers.dynamic_lstm(
    input=forward_proj, size=hidden_dim * 4, use_peepholes=False)
forward.shape  # (-1, 512)
cell.shape  # (-1, 512)