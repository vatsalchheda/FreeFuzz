import paddle

x = paddle.to_tensor([1, 2, 3, 4], dtype='float32')
y = paddle.to_tensor([2, 2, 1, 3], dtype='float32')
result = paddle.less_than(x, y)
print(result) # [True, False, False, False]