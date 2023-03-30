import paddle

x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
repeats  = paddle.to_tensor([3, 2, 1], dtype='int32')

paddle.repeat_interleave(x, repeats, 1)
# [[1, 1, 1, 2, 2, 3],
#  [4, 4, 4, 5, 5, 6]]

paddle.repeat_interleave(x, 2, 0)
# [[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6]]

paddle.repeat_interleave(x, 2, None)
# [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]