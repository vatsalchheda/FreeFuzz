import paddle

x = paddle.to_tensor([2, 3, 3, 1, 5, 3])
unique = paddle.unique(x)
np_unique = unique.numpy() # [1 2 3 5]
_, indices, inverse, counts = paddle.unique(x, return_index=True, return_inverse=True, return_counts=True)
print(indices)
# Tensor(shape=[4], dtype=int64, place=Place(gpu:0), stop_gradient=True,
#        [3, 0, 1, 4])
print(inverse)
# Tensor(shape=[6], dtype=int64, place=Place(gpu:0), stop_gradient=True,
#        [1, 2, 2, 0, 3, 2])
print(counts)
# Tensor(shape=[4], dtype=int64, place=Place(gpu:0), stop_gradient=True,
#        [1, 1, 3, 1])

x = paddle.to_tensor([[2, 1, 3], [3, 0, 1], [2, 1, 3]])
unique = paddle.unique(x)
print(unique)
# Tensor(shape=[4], dtype=int64, place=Place(gpu:0), stop_gradient=True,
#        [0, 1, 2, 3])

unique = paddle.unique(x, axis=0)
print(unique)
# Tensor(shape=[2, 3], dtype=int64, place=Place(gpu:0), stop_gradient=True,
#        [[2, 1, 3],
#         [3, 0, 1]])