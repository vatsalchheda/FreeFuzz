import paddle.fluid as fluid
import paddle
paddle.enable_static()

x = fluid.data(
    name='data', shape=[None, 256, 32, 32], dtype='float32')
rois = fluid.data(
    name='rois', shape=[None, 4], dtype='float32')
rois_num = fluid.data(name='rois_num', shape=[None], dtype='int32')
align_out = fluid.layers.roi_align(input=x,
                                   rois=rois,
                                   pooled_height=7,
                                   pooled_width=7,
                                   spatial_scale=0.5,
                                   sampling_ratio=-1,
                                   rois_num=rois_num)