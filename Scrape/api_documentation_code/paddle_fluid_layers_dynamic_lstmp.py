import paddle
paddle.enable_static()
import paddle.fluid as fluid
dict_dim, emb_dim = 128, 64
data = fluid.data(name='sequence', shape=[None], dtype='int64', lod_level=1)
emb = fluid.embedding(input=data, size=[dict_dim, emb_dim])
hidden_dim, proj_dim = 512, 256
fc_out = fluid.layers.fc(input=emb, size=hidden_dim * 4,
                        act=None, bias_attr=None)
proj_out, last_c = fluid.layers.dynamic_lstmp(input=fc_out,
                                        size=hidden_dim * 4,
                                        proj_size=proj_dim,
                                        use_peepholes=False,
                                        is_reverse=True,
                                        cell_activation="tanh",
                                        proj_activation="tanh")
proj_out.shape  # (-1, 256)
last_c.shape  # (-1, 512)