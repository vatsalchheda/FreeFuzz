import paddle.fluid as fluid
import paddle
paddle.enable_static()
x = fluid.data(name='x', shape=[100, 490, 28, 28], dtype='float32')
rois = fluid.data(name='rois', shape=[None, 4], lod_level=1, dtype='float32')
pool_out = fluid.layers.psroi_pool(x, rois, 10, 1.0, 7, 7)