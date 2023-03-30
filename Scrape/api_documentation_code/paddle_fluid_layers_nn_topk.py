import paddle.fluid as fluid
import paddle.fluid.layers as layers
# set batch size=None
input = fluid.data(name="input", shape=[None, 13, 11], dtype='float32')
top5_values, top5_indices = layers.topk(input, k=5) # top5_values.shape[None, 13, 5], top5_indices.shape=[None, 13, 5]

# 1D Tensor
input1 = fluid.data(name="input1", shape=[None, 13], dtype='float32')
top5_values, top5_indices = layers.topk(input1, k=5) #top5_values.shape=[None, 5], top5_indices.shape=[None, 5]

# k=Variable
input2 = fluid.data(name="input2", shape=[None, 13, 11], dtype='float32')
vk = fluid.data(name="vk", shape=[None, 1], dtype='int32') # save k in vk.data[0]
vk_values, vk_indices = layers.topk(input2, k=vk) #vk_values.shape=[None, 13, k], vk_indices.shape=[None, 13, k]