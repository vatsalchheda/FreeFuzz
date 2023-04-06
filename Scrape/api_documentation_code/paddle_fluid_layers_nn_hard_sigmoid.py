import paddle.fluid as fluid
import paddle
paddle.enable_static()

data = fluid.layers.fill_constant(shape=[3, 2], value=0.5, dtype='float32') # [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
result = fluid.layers.hard_sigmoid(data) # [[0.6, 0.6], [0.6, 0.6], [0.6, 0.6]]