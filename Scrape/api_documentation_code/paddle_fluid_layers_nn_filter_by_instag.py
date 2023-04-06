import paddle.fluid.layers as layers
ins = layers.data(name='Ins', shape=[-1,32], lod_level=0, dtype='float64')
ins_tag = layers.data(name='Ins_tag', shape=[-1,16], lod_level=0, dtype='int64')
filter_tag = layers.data(name='Filter_tag', shape=[-1,16], dtype='int64')
out, loss_weight = layers.filter_by_instag(ins,  ins_tag,  filter_tag, True)