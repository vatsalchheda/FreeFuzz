import numpy as np
import paddle.fluid as fluid
import paddle.fluid as fluid
import paddle

paddle.enable_static()
use_gpu = False
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)

data = fluid.data(name='data', shape=[None, 1, 2, 2], dtype='float32')
input_scale = fluid.layers.create_parameter(shape=[1], dtype="float32",
                        default_initializer=fluid.initializer.Constant(2.0))
input_bias = fluid.layers.create_parameter(shape=[1],dtype="float32",
                        default_initializer=fluid.initializer.Constant(0.5))
out = fluid.layers.affine_channel(data,scale=input_scale,
                        bias=input_bias)

exe.run(fluid.default_startup_program())
test_program = fluid.default_main_program().clone(for_test=True)

[out_array] = exe.run(test_program,
                      fetch_list=out,
                      feed={'data': np.ones([1,1,2,2]).astype('float32')})
# out_array is [[[[2.5, 2.5],
#                [2.5, 2.5]]]] with shape: [1, 1, 2, 2]