import paddle

# data_x is a Tensor with shape [2, 4]
# the axis is a int element
x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                      [0.1, 0.2, 0.6, 0.7]],
                     dtype='float64', stop_gradient=False)
result1 = paddle.max(x)
result1.backward()
print(result1, x.grad)
#[0.9], [[0., 0., 0., 1.], [0., 0., 0., 0.]]

x.clear_grad()
result2 = paddle.max(x, axis=0)
result2.backward()
print(result2, x.grad)
#[0.2, 0.3, 0.6, 0.9], [[1., 1., 0., 1.], [0., 0., 1., 0.]]

x.clear_grad()
result3 = paddle.max(x, axis=-1)
result3.backward()
print(result3, x.grad)
#[0.9, 0.7], [[0., 0., 0., 1.], [0., 0., 0., 1.]]

x.clear_grad()
result4 = paddle.max(x, axis=1, keepdim=True)
result4.backward()
print(result4, x.grad)
#[[0.9], [0.7]], [[0., 0., 0., 1.], [0., 0., 0., 1.]]

# data_y is a Tensor with shape [2, 2, 2]
# the axis is list
y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
                      [[5.0, 6.0], [7.0, 8.0]]],
                     dtype='float64', stop_gradient=False)
result5 = paddle.max(y, axis=[1, 2])
result5.backward()
print(result5, y.grad)
#[4., 8.], [[[0., 0.], [0., 1.]], [[0., 0.], [0., 1.]]]

y.clear_grad()
result6 = paddle.max(y, axis=[0, 1])
result6.backward()
print(result6, y.grad)
#[7., 8.], [[[0., 0.], [0., 0.]], [[0., 0.], [1., 1.]]]