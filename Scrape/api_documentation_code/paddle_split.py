import paddle

# x is a Tensor of shape [3, 9, 5]
x = paddle.rand([3, 9, 5])

out0, out1, out2 = paddle.split(x, num_or_sections=3, axis=1)
print(out0.shape)  # [3, 3, 5]
print(out1.shape)  # [3, 3, 5]
print(out2.shape)  # [3, 3, 5]

out0, out1, out2 = paddle.split(x, num_or_sections=[2, 3, 4], axis=1)
print(out0.shape)  # [3, 2, 5]
print(out1.shape)  # [3, 3, 5]
print(out2.shape)  # [3, 4, 5]

out0, out1, out2 = paddle.split(x, num_or_sections=[2, 3, -1], axis=1)
print(out0.shape)  # [3, 2, 5]
print(out1.shape)  # [3, 3, 5]
print(out2.shape)  # [3, 4, 5]

# axis is negative, the real axis is (rank(x) + axis)=1
out0, out1, out2 = paddle.split(x, num_or_sections=3, axis=-2)
print(out0.shape)  # [3, 3, 5]
print(out1.shape)  # [3, 3, 5]
print(out2.shape)  # [3, 3, 5]