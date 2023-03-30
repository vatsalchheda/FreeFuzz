import paddle
import numpy as np
import paddle.nn.functional as F

x_shape = (1, 1, 2, 3)
x = paddle.arange(np.prod(x_shape), dtype="float32").reshape(x_shape) + 1
y = F.zeropad2d(x, [1, 2, 1, 1])
# [[[[0. 0. 0. 0. 0. 0.]
#    [0. 1. 2. 3. 0. 0.]
#    [0. 4. 5. 6. 0. 0.]
#    [0. 0. 0. 0. 0. 0.]]]]