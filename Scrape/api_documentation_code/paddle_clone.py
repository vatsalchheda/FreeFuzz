import paddle

x = paddle.ones([2])
x.stop_gradient = False
clone_x = paddle.clone(x)

y = clone_x**3
y.backward()
print(clone_x.grad)          # [3]
print(x.grad)                # [3]