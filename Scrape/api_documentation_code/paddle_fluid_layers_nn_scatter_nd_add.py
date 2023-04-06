import paddle.fluid as fluid
import paddle
paddle.enable_static()
ref = fluid.data(name='ref', shape=[3, 5, 9, 10], dtype='float32')
index = fluid.data(name='index', shape=[3, 2], dtype='int32')
updates = fluid.data(name='update', shape=[3, 9, 10], dtype='float32')

output = fluid.layers.scatter_nd_add(ref, index, updates)