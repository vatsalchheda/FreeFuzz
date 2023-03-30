import paddle

x = paddle.to_tensor([1, 2, 3])
y = paddle.to_tensor([1, 3, 2])
result1 = paddle.less_equal(x, y)
print(result1)  # result1 = [True True False]