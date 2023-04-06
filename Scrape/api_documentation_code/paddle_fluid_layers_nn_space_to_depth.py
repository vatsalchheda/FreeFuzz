import paddle.fluid as fluid
import numpy as np
import numpy as np
import paddle

paddle.enable_static()
data = fluid.data(
    name='data', shape=[1, 4, 2, 2], dtype='float32')
space_to_depthed = fluid.layers.space_to_depth(
    x=data, blocksize=2)

exe = fluid.Executor(fluid.CPUPlace())
data_np = np.arange(0,16).reshape((1,4,2,2)).astype('float32')

print(data_np)
#array([[[[ 0.,  1.], [ 2.,  3.]],
#        [[ 4.,  5.], [ 6.,  7.]],
#        [[ 8.,  9.], [10., 11.]],
#        [[12., 13.], [14., 15.]]]], dtype=float32)

out_main = exe.run(fluid.default_main_program(),
            feed={'data': data_np},
            fetch_list=[space_to_depthed])

print(out_main)
#[array([[[[ 0.]], [[ 4.]], [[ 1.]], [[ 5.]],
#         [[ 8.]], [[12.]], [[ 9.]], [[13.]],
#         [[ 2.]], [[ 6.]], [[ 3.]], [[ 7.]],
#         [[10.]], [[14.]], [[11.]], [[15.]]]], dtype=float32)]