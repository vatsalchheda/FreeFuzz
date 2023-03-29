import paddle

x = paddle.to_tensor([1, 2, 1, 4, 5])
result1 = paddle.bincount(x)
print(result1) # [0, 2, 1, 0, 1, 1]

w = paddle.to_tensor([2.1, 0.4, 0.1, 0.5, 0.5])
result2 = paddle.bincount(x, weights=w)
print(result2) # [0., 2.19999981, 0.40000001, 0., 0.50000000, 0.50000000]