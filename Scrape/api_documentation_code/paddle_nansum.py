import paddle

# x is a Tensor with following elements:
#    [[nan, 0.3, 0.5, 0.9]
#     [0.1, 0.2, -nan, 0.7]]
# Each example is followed by the corresponding output tensor.
x = paddle.to_tensor([[float('nan'), 0.3, 0.5, 0.9],
                [0.1, 0.2, float('-nan'), 0.7]],dtype="float32")
out1 = paddle.nansum(x)  # [2.7]
out2 = paddle.nansum(x, axis=0)  # [0.1, 0.5, 0.5, 1.6]
out3 = paddle.nansum(x, axis=-1)  # [1.7, 1.0]
out4 = paddle.nansum(x, axis=1, keepdim=True)  # [[1.7], [1.0]]

# y is a Tensor with shape [2, 2, 2] and elements as below:
#      [[[1, nan], [3, 4]],
#      [[5, 6], [-nan, 8]]]
# Each example is followed by the corresponding output tensor.
y = paddle.to_tensor([[[1, float('nan')], [3, 4]],
                [[5, 6], [float('-nan'), 8]]])
out5 = paddle.nansum(y, axis=[1, 2]) # [8, 19]
out6 = paddle.nansum(y, axis=[0, 1]) # [9, 18]