import numpy as np
import paddle
import paddle.nn.functional as F

# example 1
x_shape = (1, 1, 3, 4)
x = np.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape) + 1
tensor_x = paddle.to_tensor(x)
y = paddle.fluid.layers.pad2d(tensor_x, paddings=[1, 2, 2, 1], pad_value=1, mode='constant')
print(y.numpy())
# [[[[ 1.  1.  1.  1.  1.  1.  1.]
#    [ 1.  1.  1.  2.  3.  4.  1.]
#    [ 1.  1.  5.  6.  7.  8.  1.]
#    [ 1.  1.  9. 10. 11. 12.  1.]
#    [ 1.  1.  1.  1.  1.  1.  1.]
#    [ 1.  1.  1.  1.  1.  1.  1.]]]]

# example 2
x_shape = (1, 1, 2, 3)
x = np.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape) + 1
tensor_x = paddle.to_tensor(x)
y = paddle.fluid.layers.pad2d(tensor_x, paddings=[1, 1, 1, 1], mode='reflect')
print(y.numpy())
# [[[[5. 4. 5. 6. 5.]
#    [2. 1. 2. 3. 2.]
#    [5. 4. 5. 6. 5.]
#    [2. 1. 2. 3. 2.]]]]