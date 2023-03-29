import paddle

data = paddle.to_tensor([1, 2, 3], dtype='int32')
out = paddle.broadcast_to(data, shape=[2, 3])
print(out)
# [[1, 2, 3], [1, 2, 3]]