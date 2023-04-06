#deformable conv v2:

import paddle.fluid as fluid
import paddle
paddle.enable_static()

C_in, H_in, W_in = 3, 32, 32
filter_size, deformable_groups = 3, 1
data = fluid.data(name='data', shape=[None, C_in, H_in, W_in], dtype='float32')
offset = fluid.data(name='offset', shape=[None, 2*deformable_groups*filter_size**2, H_in, W_in], dtype='float32')
mask = fluid.data(name='mask', shape=[None, deformable_groups*filter_size**2, H_in, W_in], dtype='float32')
out = fluid.layers.deformable_conv(input=data, offset=offset, mask=mask,
                                   num_filters=2, filter_size=filter_size, padding=1, modulated=True)

#deformable conv v1:

import paddle.fluid as fluid
C_in, H_in, W_in = 3, 32, 32
filter_size, deformable_groups = 3, 1
data = fluid.data(name='data', shape=[None, C_in, H_in, W_in], dtype='float32')
offset = fluid.data(name='offset', shape=[None, 2*deformable_groups*filter_size**2, H_in, W_in], dtype='float32')
out = fluid.layers.deformable_conv(input=data, offset=offset, mask=None,
                                   num_filters=2, filter_size=filter_size, padding=1, modulated=False)