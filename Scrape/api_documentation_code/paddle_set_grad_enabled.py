import paddle
x = paddle.ones([3, 2])
x.stop_gradient = False
with paddle.set_grad_enabled(False):
    y = x * 2
    with paddle.set_grad_enabled(True):
        z = x * 2
print(y.stop_gradient)   # True
print(z.stop_gradient)   # False