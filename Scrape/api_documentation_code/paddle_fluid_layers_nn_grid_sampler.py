import paddle.fluid as fluid
import paddle.fluid as fluid
import paddle

paddle.enable_static()
# use with affine_grid
x = fluid.data(name='x', shape=[None, 10, 32, 32], dtype='float32')
theta = fluid.layers.data(name='theta', shape=[2, 3], dtype='float32')
grid = fluid.layers.affine_grid(theta=theta, out_shape=[3, 10, 32, 32])
out = fluid.layers.grid_sampler(x=x, grid=grid)