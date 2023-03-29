import paddle

sorted_sequence = paddle.to_tensor([[1, 3, 5, 7, 9, 11],
                                    [2, 4, 6, 8, 10, 12]], dtype='int32')
values = paddle.to_tensor([[3, 6, 9, 10], [3, 6, 9, 10]], dtype='int32')
out1 = paddle.searchsorted(sorted_sequence, values)
print(out1)
# Tensor(shape=[2, 4], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#        [[1, 3, 4, 5],
#         [1, 2, 4, 4]])
out2 = paddle.searchsorted(sorted_sequence, values, right=True)
print(out2)
# Tensor(shape=[2, 4], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#        [[2, 3, 5, 5],
#         [1, 3, 4, 5]])
sorted_sequence_1d = paddle.to_tensor([1, 3, 5, 7, 9, 11, 13])
out3 = paddle.searchsorted(sorted_sequence_1d, values)
print(out3)
# Tensor(shape=[2, 4], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
#        [[1, 3, 4, 5],
#         [1, 3, 4, 5]])