import paddle

x = paddle.to_tensor([1, 1, 2, 2, 3, 1, 1, 2])
output = paddle.unique_consecutive(x) #
print(output)
# Tensor(shape=[5], dtype=int64, place=Place(gpu:0), stop_gradient=True,
#        [1, 2, 3, 1, 2])

_, inverse, counts = paddle.unique_consecutive(x, return_inverse=True, return_counts=True)
print(inverse)
# Tensor(shape=[8], dtype=int64, place=Place(gpu:0), stop_gradient=True,
#        [0, 0, 1, 1, 2, 3, 3, 4])
print(counts)
# Tensor(shape=[5], dtype=int64, place=Place(gpu:0), stop_gradient=True,
#        [2, 2, 1, 2, 1])

x = paddle.to_tensor([[2, 1, 3], [3, 0, 1], [2, 1, 3], [2, 1, 3]])
output = paddle.unique_consecutive(x, axis=0) #
print(output)
# Tensor(shape=[3, 3], dtype=int64, place=Place(gpu:0), stop_gradient=True,
#        [[2, 1, 3],
#         [3, 0, 1],
#         [2, 1, 3]])

x = paddle.to_tensor([[2, 1, 3], [3, 0, 1], [2, 1, 3], [2, 1, 3]])
output = paddle.unique_consecutive(x, axis=0) #
print(output)
# Tensor(shape=[3, 3], dtype=int64, place=Place(gpu:0), stop_gradient=True,
#        [[2, 1, 3],
#         [3, 0, 1],
#         [2, 1, 3]])