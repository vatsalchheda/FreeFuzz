import paddle

data = paddle.arange(12)
data = paddle.reshape(data, (3, 4))
# [[ 0  1  2  3 ]
#  [ 4  5  6  7 ]
#  [ 8  9  10 11]]

y = paddle.cumprod(data, dim=0)
# [[ 0  1   2   3]
#  [ 0  5  12  21]
#  [ 0 45 120 231]]

y = paddle.cumprod(data, dim=-1)
# [[ 0   0   0    0]
#  [ 4  20 120  840]
#  [ 8  72 720 7920]]

y = paddle.cumprod(data, dim=1, dtype='float64')
# [[ 0.   0.   0.    0.]
#  [ 4.  20. 120.  840.]
#  [ 8.  72. 720. 7920.]]

print(y.dtype)
# paddle.float64