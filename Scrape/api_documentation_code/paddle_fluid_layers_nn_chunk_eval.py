import paddle
paddle.enable_static()
import paddle.fluid as fluid

dict_size = 10000
label_dict_len = 7
sequence = fluid.data(
    name='id', shape=[None, 1], lod_level=1, dtype='int64')
embedding = fluid.embedding(
    input=sequence, size=[dict_size, 512])
hidden = fluid.layers.fc(input=embedding, size=512)
label = fluid.data(
    name='label', shape=[None, 1], lod_level=1, dtype='int64')
crf = fluid.layers.linear_chain_crf(
    input=hidden, label=label, param_attr=fluid.ParamAttr(name="crfw"))
crf_decode = fluid.layers.crf_decoding(
    input=hidden, param_attr=fluid.ParamAttr(name="crfw"))
fluid.layers.chunk_eval(
    input=crf_decode,
    label=label,
    chunk_scheme="IOB",
    num_chunk_types=int((label_dict_len - 1) / 2))