import paddle.fluid as fluid
import paddle
paddle.enable_static()
img = fluid.data(name='img', shape=[100, 1, 28, 28], dtype='float32')
conv_pool = fluid.nets.simple_img_conv_pool(input=img,
                                            filter_size=5,
                                            num_filters=20,
                                            pool_size=2,
                                            pool_stride=2,
                                            act="relu")