import paddle.fluid as fluid
import paddle

paddle.enable_static()
conv1 = fluid.data(name='conv1', shape=[None, 48, 16, 16], dtype='float32')
anchor, var = fluid.layers.anchor_generator(
    input=conv1,
    anchor_sizes=[64, 128, 256, 512],
    aspect_ratios=[0.5, 1.0, 2.0],
    variance=[0.1, 0.1, 0.2, 0.2],
    stride=[16.0, 16.0],
    offset=0.5)