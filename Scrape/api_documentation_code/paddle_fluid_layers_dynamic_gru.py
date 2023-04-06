import paddle.fluid as fluid

dict_dim, emb_dim = 128, 64
data = fluid.data(name='sequence',
          shape=[None],
          dtype='int64',
          lod_level=1)
emb = fluid.embedding(input=data, size=[dict_dim, emb_dim])
hidden_dim = 512
x = fluid.layers.fc(input=emb, size=hidden_dim * 3)
hidden = fluid.layers.dynamic_gru(input=x, size=hidden_dim)