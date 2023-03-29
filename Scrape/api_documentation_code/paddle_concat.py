import paddle

x1 = paddle.to_tensor([[1, 2, 3],
                       [4, 5, 6]])
x2 = paddle.to_tensor([[11, 12, 13],
                       [14, 15, 16]])
x3 = paddle.to_tensor([[21, 22],
                       [23, 24]])
zero = paddle.full(shape=[1], dtype='int32', fill_value=0)
# When the axis is negative, the real axis is (axis + Rank(x))
# As follow, axis is -1, Rank(x) is 2, the real axis is 1
out1 = paddle.concat(x=[x1, x2, x3], axis=-1)
out2 = paddle.concat(x=[x1, x2], axis=0)
out3 = paddle.concat(x=[x1, x2], axis=zero)
# out1
# [[ 1  2  3 11 12 13 21 22]
#  [ 4  5  6 14 15 16 23 24]]
# out2 out3
# [[ 1  2  3]
#  [ 4  5  6]
#  [11 12 13]
#  [14 15 16]]