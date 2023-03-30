import paddle.fluid as fluid
import paddle
paddle.enable_static()
input_dim = 100 #len(word_dict)
emb_dim = 128
hid_dim = 512
data = fluid.data(name="words", shape=[None, 1], dtype="int64", lod_level=1)
emb = fluid.layers.embedding(input=data, size=[input_dim, emb_dim], is_sparse=True)
seq_conv = fluid.nets.sequence_conv_pool(input=emb,
                                         num_filters=hid_dim,
                                         filter_size=3,
                                         act="tanh",
                                         pool_type="sqrt")