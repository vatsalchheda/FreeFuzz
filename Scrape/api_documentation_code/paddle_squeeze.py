import paddle

x = paddle.rand([5, 1, 10])
output = paddle.squeeze(x, axis=1)

print(x.shape)  # [5, 1, 10]
print(output.shape)  # [5, 10]

# output shares data with x in dygraph mode
x[0, 0, 0] = 10.
print(output[0, 0]) # [10.]