import paddle

x = paddle.to_tensor([1, 2, 3])
y = paddle.to_tensor([1, 2, 3])
z = paddle.to_tensor([1, 4, 3])
result1 = paddle.equal_all(x, y)
print(result1) # result1 = [True ]
result2 = paddle.equal_all(x, z)
print(result2) # result2 = [False ]