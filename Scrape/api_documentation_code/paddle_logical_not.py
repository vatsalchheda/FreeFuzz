import paddle

x = paddle.to_tensor([True, False, True, False])
res = paddle.logical_not(x)
print(res) # [False  True False  True]