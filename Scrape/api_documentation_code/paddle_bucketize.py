import paddle

sorted_sequence = paddle.to_tensor([2, 4, 8, 16], dtype='int32')
x = paddle.to_tensor([[0, 8, 4, 16], [-1, 2, 8, 4]], dtype='int32')
out1 = paddle.bucketize(x, sorted_sequence)
print(out1)
# Tensor(shape=[2, 4], dtype=int64, place=CPUPlace, stop_gradient=True,
#        [[0, 2, 1, 3],
#         [0, 0, 2, 1]])
out2 = paddle.bucketize(x, sorted_sequence, right=True)
print(out2)
# Tensor(shape=[2, 4], dtype=int64, place=CPUPlace, stop_gradient=True,
#        [[0, 3, 2, 4],
#         [0, 1, 3, 2]])
out3 = x.bucketize(sorted_sequence)
print(out3)
# Tensor(shape=[2, 4], dtype=int64, place=CPUPlace, stop_gradient=True,
#        [[0, 2, 1, 3],
#         [0, 0, 2, 1]])
out4 = x.bucketize(sorted_sequence, right=True)
print(out4)
# Tensor(shape=[2, 4], dtype=int64, place=CPUPlace, stop_gradient=True,
#        [[0, 3, 2, 4],
#         [0, 1, 3, 2]])