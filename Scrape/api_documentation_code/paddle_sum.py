import paddle

# x is a Tensor with following elements:
#    [[0.2, 0.3, 0.5, 0.9]
#     [0.1, 0.2, 0.6, 0.7]]
# Each example is followed by the corresponding output tensor.
x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                      [0.1, 0.2, 0.6, 0.7]])
out1 = paddle.sum(x)  # [3.5]
out2 = paddle.sum(x, axis=0)  # [0.3, 0.5, 1.1, 1.6]
out3 = paddle.sum(x, axis=-1)  # [1.9, 1.6]
out4 = paddle.sum(x, axis=1, keepdim=True)  # [[1.9], [1.6]]

# y is a Tensor with shape [2, 2, 2] and elements as below:
#      [[[1, 2], [3, 4]],
#      [[5, 6], [7, 8]]]
# Each example is followed by the corresponding output tensor.
y = paddle.to_tensor([[[1, 2], [3, 4]],
                      [[5, 6], [7, 8]]])
out5 = paddle.sum(y, axis=[1, 2]) # [10, 26]
out6 = paddle.sum(y, axis=[0, 1]) # [16, 20]

# x is a Tensor with following elements:
#    [[True, True, True, True]
#     [False, False, False, False]]
# Each example is followed by the corresponding output tensor.
x = paddle.to_tensor([[True, True, True, True],
                      [False, False, False, False]])
out7 = paddle.sum(x)  # [4]
out8 = paddle.sum(x, axis=0)  # [1, 1, 1, 1]
out9 = paddle.sum(x, axis=1)  # [4, 0]