import paddle.fluid as fluid
import numpy as np
import paddle
paddle.enable_static()

data = fluid.data(name='x', shape=[None, 1], dtype='int64')

# example 1
emb_1 = fluid.embedding(input=data, size=[128, 64])

# example 2: load custom or pre-trained word vectors
weight_data = np.random.random(size=(128, 100))  # word vectors with numpy format
w_param_attrs = fluid.ParamAttr(
    name="emb_weight",
    learning_rate=0.5,
    initializer=fluid.initializer.NumpyArrayInitializer(weight_data),
    trainable=True)
emb_2 = fluid.layers.embedding(input=data, size=(128, 100), param_attr=w_param_attrs, dtype='float32')