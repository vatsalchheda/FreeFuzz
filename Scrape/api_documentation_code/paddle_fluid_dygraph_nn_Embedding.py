import paddle.fluid as fluid
import paddle.fluid.dygraph.base as base
import numpy as np

# example 1
inp_word = np.array([[2, 3, 5], [4, 2, 1]]).astype('int64')
inp_word.shape  # [2, 3]
dict_size = 20
with fluid.dygraph.guard():
    emb = fluid.dygraph.Embedding(
        size=[dict_size, 32],
        param_attr='emb.w',
        is_sparse=False)
    static_rlt3 = emb(base.to_variable(inp_word))
    static_rlt3.shape  # [2, 3, 32]

# example 2: load custom or pre-trained word vectors
weight_data = np.random.random(size=(128, 100))  # word vectors with numpy format
w_param_attrs = fluid.ParamAttr(
    name="emb_weight",
    learning_rate=0.5,
    initializer=fluid.initializer.NumpyArrayInitializer(weight_data),
    trainable=True)
with fluid.dygraph.guard():
    emb = fluid.dygraph.Embedding(
        size=[128, 100],
        param_attr= w_param_attrs,
        is_sparse=False)
    static_rlt3 = emb(base.to_variable(inp_word))