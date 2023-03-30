import paddle
from paddle.jit import to_static

@to_static
def func(x):
    if paddle.mean(x) < 0:
        x_v = x - 1
    else:
        x_v = x + 1
    return x_v

x = paddle.ones([1, 2], dtype='float32')
x_v = func(x)
print(x_v) # [[2. 2.]]