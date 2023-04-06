import paddle.fluid as fluid
import paddle
paddle.enable_static()
boxes = fluid.data(
    name='boxes', shape=[None, 8, 4], dtype='float32', lod_level=1)
im_info = fluid.data(name='im_info', shape=[-1 ,3])
out = fluid.layers.box_clip(
    input=boxes, im_info=im_info)