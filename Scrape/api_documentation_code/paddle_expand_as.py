import paddle

data_x = paddle.to_tensor([1, 2, 3], 'int32')
data_y = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], 'int32')
out = paddle.expand_as(data_x, data_y)
print(out)
# Tensor(shape=[2, 3], dtype=int32, place=Place(gpu:0), stop_gradient=True,
#        [[1, 2, 3],
#         [1, 2, 3]])