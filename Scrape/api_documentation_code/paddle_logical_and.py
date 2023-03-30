import paddle

x = paddle.to_tensor([True])
y = paddle.to_tensor([True, False, True, False])
res = paddle.logical_and(x, y)
print(res) # [True False True False]