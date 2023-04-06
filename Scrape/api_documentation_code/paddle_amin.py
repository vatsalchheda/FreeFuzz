import paddle
# data_x is a Tensor with shape [2, 4] with multiple minimum elements
# the axis is a int element

x = paddle.to_tensor([[0.2, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.6, 0.7]],
                     dtype='float64', stop_gradient=False)
# There are 5 minimum elements:
# 1) amin evenly distributes gradient between these equal values,
#    thus the corresponding gradients are 1/5=0.2;
# 2) while min propagates gradient to all of them,
#    thus the corresponding gradient are 1.
result1 = paddle.amin(x)
result1.backward()
print(result1, x.grad)
#[0.1], [[0., 0.2, 0.2, 0.2], [0.2, 0.2, 0., 0.]]

x.clear_grad()
result1_min = paddle.min(x)
result1_min.backward()
print(result1_min, x.grad)
#[0.1], [[0., 1.0, 1.0, 1.0], [1.0, 1.0, 0., 0.]]

###############################

x.clear_grad()
result2 = paddle.amin(x, axis=0)
result2.backward()
print(result2, x.grad)
#[0.1, 0.1, 0.1, 0.1], [[0., 0.5, 1., 1.], [1., 0.5, 0., 0.]]

x.clear_grad()
result3 = paddle.amin(x, axis=-1)
result3.backward()
print(result3, x.grad)
#[0.1, 0.1], [[0., 0.3333, 0.3333, 0.3333], [0.5, 0.5, 0., 0.]]

x.clear_grad()
result4 = paddle.amin(x, axis=1, keepdim=True)
result4.backward()
print(result4, x.grad)
#[[0.1], [0.1]], [[0., 0.3333, 0.3333, 0.3333.], [0.5, 0.5, 0., 0.]]

# data_y is a Tensor with shape [2, 2, 2]
# the axis is list
y = paddle.to_tensor([[[0.2, 0.1], [0.1, 0.1]],
                      [[0.1, 0.1], [0.6, 0.7]]],
                     dtype='float64', stop_gradient=False)
result5 = paddle.amin(y, axis=[1, 2])
result5.backward()
print(result5, y.grad)
#[0.1., 0.1], [[[0., 0.3333], [0.3333, 0.3333]], [[0.5, 0.5], [0., 1.]]]

y.clear_grad()
result6 = paddle.amin(y, axis=[0, 1])
result6.backward()
print(result6, y.grad)
#[0.1., 0.1], [[[0., 0.3333], [0.5, 0.3333]], [[0.5, 0.3333], [1., 1.]]]