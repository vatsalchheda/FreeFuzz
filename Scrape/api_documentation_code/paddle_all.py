import paddle

# x is a bool Tensor with following elements:
#    [[True, False]
#     [True, True]]
x = paddle.to_tensor([[1, 0], [1, 1]], dtype='int32')
print(x)
x = paddle.cast(x, 'bool')

# out1 should be [False]
out1 = paddle.all(x)  # [False]
print(out1)

# out2 should be [True, False]
out2 = paddle.all(x, axis=0)  # [True, False]
print(out2)

# keepdim=False, out3 should be [False, True], out.shape should be (2,)
out3 = paddle.all(x, axis=-1)  # [False, True]
print(out3)

# keepdim=True, out4 should be [[False], [True]], out.shape should be (2,1)
out4 = paddle.all(x, axis=1, keepdim=True) # [[False], [True]]
print(out4)