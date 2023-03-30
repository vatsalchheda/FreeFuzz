# x is a rank 4 tensor variable, x.shape = (2, 3, 2, 3)
# y is a rank 4 tensor variable, y.shape = (1, 3, 1, 3)
import paddle.fluid as fluid
x = fluid.data(name='x', shape=[2,3,2,3], dtype='float32')
y = fluid.data(name='y', shape=[1,3,1,3], dtype='float32')
out = fluid.layers.pad_constant_like(x=x, y=y, pad_value=0.)
# out is a rank 4 tensor variable, and out.shape = [2, 3 ,2 , 3]