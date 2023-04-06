import paddle

x1 = paddle.to_tensor([[1.0, 2.0]])
x2 = paddle.to_tensor([[3.0, 4.0]])
x3 = paddle.to_tensor([[5.0, 6.0]])

out = paddle.stack([x1, x2, x3], axis=0)
print(out.shape)  # [3, 1, 2]
print(out)
# [[[1., 2.]],
#  [[3., 4.]],
#  [[5., 6.]]]

out = paddle.stack([x1, x2, x3], axis=-2)
print(out.shape)  # [1, 3, 2]
print(out)
# [[[1., 2.],
#   [3., 4.],
#   [5., 6.]]]