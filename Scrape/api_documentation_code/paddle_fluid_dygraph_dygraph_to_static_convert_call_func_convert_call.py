import paddle
from paddle.jit.dy2static import convert_call

paddle.enable_static()
def dyfunc(x):
    if paddle.mean(x) < 0:
        x_v = x - 1
    else:
        x_v = x + 1
    return x_v

new_func = convert_call(dyfunc)
x = paddle.tensor.manipulation.fill_constant(shape=[3, 3], value=0, dtype='float64')
x_v = new_func(x)

exe = paddle.static.Executor(paddle.CPUPlace())
out = exe.run(fetch_list=[x_v])
print(out[0])
# [[1. 1. 1.]
#  [1. 1. 1.]
#  [1. 1. 1.]]