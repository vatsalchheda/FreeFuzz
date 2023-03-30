import paddle

# data_x is a Tensor with shape [2, 4]
# the axis is a int element
x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                      [0.1, 0.2, 0.6, 0.7]],
                     dtype='float64', stop_gradient=False)
result1 = paddle.min(x)
result1.backward()
print(result1, x.grad)
#[0.1], [[0., 0., 0., 0.], [1., 0., 0., 0.]]

x.clear_grad()
result2 = paddle.min(x, axis=0)
result2.backward()
print(result2, x.grad)
#[0.1, 0.2, 0.5, 0.7], [[0., 0., 1., 0.], [1., 1., 0., 1.]]

x.clear_grad()
result3 = paddle.min(x, axis=-1)
result3.backward()
print(result3, x.grad)
#[0.2, 0.1], [[1., 0., 0., 0.], [1., 0., 0., 0.]]

x.clear_grad()
result4 = paddle.min(x, axis=1, keepdim=True)
result4.backward()
print(result4, x.grad)
#[[0.2], [0.1]], [[1., 0., 0., 0.], [1., 0., 0., 0.]]

# data_y is a Tensor with shape [2, 2, 2]
# the axis is list
y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
                      [[5.0, 6.0], [7.0, 8.0]]],
                     dtype='float64', stop_gradient=False)
result5 = paddle.min(y, axis=[1, 2])
result5.backward()
print(result5, y.grad)
#[1., 5.], [[[1., 0.], [0., 0.]], [[1., 0.], [0., 0.]]]

y.clear_grad()
result6 = paddle.min(y, axis=[0, 1])
result6.backward()
print(result6, y.grad)
#[1., 2.], [[[1., 1.], [0., 0.]], [[0., 0.], [0., 0.]]]