import numpy as np
import paddle

data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
with paddle.fluid.dygraph.guard():
    conv2d = paddle.fluid.dygraph.Conv2D(3, 2, 3)
    data = paddle.fluid.dygraph.to_variable(data)
    with paddle.fluid.dygraph.amp_guard():
        conv = conv2d(data)
        print(conv.dtype) # FP16
    with paddle.fluid.dygraph.amp_guard(enable=False):
        conv = conv2d(data)
        print(conv.dtype) # FP32