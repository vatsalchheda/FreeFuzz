import paddle

data = paddle.arange(12)
data = paddle.reshape(data, (3, 4))

y = paddle.cumsum(data)
# [ 0  1  3  6 10 15 21 28 36 45 55 66]

y = paddle.cumsum(data, axis=0)
# [[ 0  1  2  3]
#  [ 4  6  8 10]
#  [12 15 18 21]]

y = paddle.cumsum(data, axis=-1)
# [[ 0  1  3  6]
#  [ 4  9 15 22]
#  [ 8 17 27 38]]

y = paddle.cumsum(data, dtype='float64')
print(y.dtype)
# paddle.float64