# Examples to clarify shapes of the inputs and output
# x: [B, ..., M, K], y: [B, ..., K, N]
# fluid.layers.matmul(x, y)  # out: [B, ..., M, N]

# x: [B, M, K], y: [B, K, N]
# fluid.layers.matmul(x, y)  # out: [B, M, N]

# x: [B, M, K], y: [K, N]
# fluid.layers.matmul(x, y)  # out: [B, M, N]

# x: [M, K], y: [K, N]
# fluid.layers.matmul(x, y)  # out: [M, N]

# x: [B, M, K], y: [K]
# fluid.layers.matmul(x, y)  # out: [B, M]

# x: [K], y: [K]
# fluid.layers.matmul(x, y)  # out: [1]

# x: [M], y: [N]
# fluid.layers.matmul(x, y, True, True)  # out: [M, N]

import paddle
import paddle.fluid as fluid
paddle.enable_static()

x = fluid.layers.data(name='x', shape=[2, 3], dtype='float32')
y = fluid.layers.data(name='y', shape=[3, 2], dtype='float32')
out = fluid.layers.matmul(x, y, True, True)