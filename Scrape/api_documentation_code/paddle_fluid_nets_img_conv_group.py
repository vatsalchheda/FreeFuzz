import paddle.fluid as fluid
import paddle
paddle.enable_static()

img = fluid.data(name='img', shape=[None, 1, 28, 28], dtype='float32')
conv_pool = fluid.nets.img_conv_group(input=img,
                                      conv_padding=1,
                                      conv_num_filter=[3, 3],
                                      conv_filter_size=3,
                                      conv_act="relu",
                                      pool_size=2,
                                      pool_stride=2)