import paddle

x = paddle.to_tensor([[1, 1], [2, 2], [3, 3]], dtype='float32')
index = paddle.to_tensor([2, 1, 0, 1], dtype='int64')
updates = paddle.to_tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype='float32')

output1 = paddle.scatter(x, index, updates, overwrite=False)
# [[3., 3.],
#  [6., 6.],
#  [1., 1.]]

output2 = paddle.scatter(x, index, updates, overwrite=True)
# CPU device:
# [[3., 3.],
#  [4., 4.],
#  [1., 1.]]
# GPU device maybe have two results because of the repeated numbers in index
# result 1:
# [[3., 3.],
#  [4., 4.],
#  [1., 1.]]
# result 2:
# [[3., 3.],
#  [2., 2.],
#  [1., 1.]]