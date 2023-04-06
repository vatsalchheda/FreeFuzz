import paddle

x = paddle.to_tensor([[1, 0], [1, 1]], dtype='int32')
x = paddle.assign(x)
print(x)
x = paddle.cast(x, 'bool')
# x is a bool Tensor with following elements:
#    [[True, False]
#     [True, True]]

# out1 should be [True]
out1 = paddle.any(x)  # [True]
print(out1)

# out2 should be [True, True]
out2 = paddle.any(x, axis=0)  # [True, True]
print(out2)

# keepdim=False, out3 should be [True, True], out.shape should be (2,)
out3 = paddle.any(x, axis=-1)  # [True, True]
print(out3)

# keepdim=True, result should be [[True], [True]], out.shape should be (2,1)
out4 = paddle.any(x, axis=1, keepdim=True)  # [[True], [True]]
print(out4)