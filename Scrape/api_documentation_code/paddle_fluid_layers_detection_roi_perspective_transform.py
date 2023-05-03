import paddle
paddle.enable_static()
import paddle.fluid as fluid

x = fluid.data(name='x', shape=[100, 256, 28, 28], dtype='float32')
rois = fluid.data(name='rois', shape=[None, 8], lod_level=1, dtype='float32')
out, mask, transform_matrix = fluid.layers.roi_perspective_transform(x, rois, 7, 7, 1.0)