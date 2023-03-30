import paddle
import paddle.nn.functional as F

# example 1
x_shape = (1, 1, 3)
x = paddle.arange(paddle.prod(paddle.to_tensor(x_shape)), dtype="float32").reshape(x_shape) + 1
y = F.pad(x, [0, 0, 0, 0, 2, 3], value=1, mode='constant', data_format="NCL")
print(y)
# [[[1. 1. 1. 2. 3. 1. 1. 1.]]]

# example 2
x_shape = (1, 1, 3)
x = paddle.arange(paddle.prod(paddle.to_tensor(x_shape)), dtype="float32").reshape(x_shape) + 1
y = F.pad(x, [2, 3], value=1, mode='constant', data_format="NCL")
print(y)
# [[[1. 1. 1. 2. 3. 1. 1. 1.]]]

# example 3
x_shape = (1, 1, 2, 3)
x = paddle.arange(paddle.prod(paddle.to_tensor(x_shape)), dtype="float32").reshape(x_shape) + 1
y = F.pad(x, [1, 2, 1, 1], value=1, mode='circular')
print(y)
# [[[[6. 4. 5. 6. 4. 5.]
#    [3. 1. 2. 3. 1. 2.]
#    [6. 4. 5. 6. 4. 5.]
#    [3. 1. 2. 3. 1. 2.]]]]