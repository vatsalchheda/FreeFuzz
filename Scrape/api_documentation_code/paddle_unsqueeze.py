import paddle

x = paddle.rand([5, 10])
print(x.shape)  # [5, 10]

out1 = paddle.unsqueeze(x, axis=0)
print(out1.shape)  # [1, 5, 10]

out2 = paddle.unsqueeze(x, axis=[0, 2])
print(out2.shape)  # [1, 5, 1, 10]

axis = paddle.to_tensor([0, 1, 2])
out3 = paddle.unsqueeze(x, axis=axis)
print(out3.shape)  # [1, 1, 1, 5, 10]

# out1, out2, out3 share data with x in dygraph mode
x[0, 0] = 10.
print(out1[0, 0, 0]) # [10.]
print(out2[0, 0, 0, 0]) # [10.]
print(out3[0, 0, 0, 0, 0]) # [10.]