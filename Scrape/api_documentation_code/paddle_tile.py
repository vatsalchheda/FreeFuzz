import paddle

data = paddle.to_tensor([1, 2, 3], dtype='int32')
out = paddle.tile(data, repeat_times=[2, 1])
print(out)
# Tensor(shape=[2, 3], dtype=int32, place=Place(gpu:0), stop_gradient=True,
#        [[1, 2, 3],
#         [1, 2, 3]])

out = paddle.tile(data, repeat_times=(2, 2))
print(out)
# Tensor(shape=[2, 6], dtype=int32, place=Place(gpu:0), stop_gradient=True,
#        [[1, 2, 3, 1, 2, 3],
#         [1, 2, 3, 1, 2, 3]])

repeat_times = paddle.to_tensor([1, 2], dtype='int32')
out = paddle.tile(data, repeat_times=repeat_times)
print(out)
# Tensor(shape=[1, 6], dtype=int32, place=Place(gpu:0), stop_gradient=True,
#        [[1, 2, 3, 1, 2, 3]])