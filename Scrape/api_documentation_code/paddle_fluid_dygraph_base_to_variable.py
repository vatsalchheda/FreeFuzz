import numpy as np
import paddle.fluid as fluid

with fluid.dygraph.guard(fluid.CPUPlace()):
    x = np.ones([2, 2], np.float32)
    y = fluid.dygraph.to_variable(x, zero_copy=False)
    x[0][0] = -1
    y[0][0].numpy()  # array([1.], dtype=float32)
    y = fluid.dygraph.to_variable(x)
    x[0][0] = 0
    y[0][0].numpy()  # array([0.], dtype=float32)
    c = np.array([2+1j, 2])
    z = fluid.dygraph.to_variable(c)
    z.numpy() # array([2.+1.j, 2.+0.j])
    z.dtype # 'complex128'

    y = fluid.dygraph.to_variable([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
    y.shape     # [3L, 2L]

    y = fluid.dygraph.to_variable(((0.1, 1.2), (2.2, 3.1), (4.9, 5.2)), dtype='int32')
    y.shape     # [3L, 2L]