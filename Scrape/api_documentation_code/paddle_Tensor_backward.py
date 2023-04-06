import paddle
x = paddle.to_tensor(5., stop_gradient=False)
for i in range(5):
    y = paddle.pow(x, 4.0)
    y.backward()
    print("{}: {}".format(i, x.grad))
# 0: [500.]
# 1: [1000.]
# 2: [1500.]
# 3: [2000.]
# 4: [2500.]

x.clear_grad()
print("{}".format(x.grad))
# 0.

grad_tensor=paddle.to_tensor(2.)
for i in range(5):
    y = paddle.pow(x, 4.0)
    y.backward(grad_tensor)
    print("{}: {}".format(i, x.grad))
# 0: [1000.]
# 1: [2000.]
# 2: [3000.]
# 3: [4000.]
# 4: [5000.]