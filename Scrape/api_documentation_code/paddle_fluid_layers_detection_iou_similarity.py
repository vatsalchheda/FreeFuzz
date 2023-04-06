import numpy as np
import paddle.fluid as fluid

use_gpu = False
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)

x = fluid.data(name='x', shape=[None, 4], dtype='float32')
y = fluid.data(name='y', shape=[None, 4], dtype='float32')
iou = fluid.layers.iou_similarity(x=x, y=y)

exe.run(fluid.default_startup_program())
test_program = fluid.default_main_program().clone(for_test=True)

[out_iou] = exe.run(test_program,
        fetch_list=iou,
        feed={'x': np.array([[0.5, 0.5, 2.0, 2.0],
                             [0., 0., 1.0, 1.0]]).astype('float32'),
              'y': np.array([[1.0, 1.0, 2.5, 2.5]]).astype('float32')})
# out_iou is [[0.2857143],
#             [0.       ]] with shape: [2, 1]