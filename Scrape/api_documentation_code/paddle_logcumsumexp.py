import paddle

data = paddle.arange(12, dtype='float64')
data = paddle.reshape(data, (3, 4))

y = paddle.logcumsumexp(data)
# [ 0.         1.3132617  2.4076061  3.4401898  4.4519143  5.4561934
#   6.4577627  7.4583397  8.458551   9.45863   10.458658  11.458669 ]

y = paddle.logcumsumexp(data, axis=0)
# [[ 0.        1.        2.        3.      ]
#  [ 4.01815   5.01815   6.01815   7.01815 ]
#  [ 8.018479  9.018479 10.018479 11.018479]]

y = paddle.logcumsumexp(data, axis=-1)
# [[ 0.         1.3132617  2.4076061  3.4401898]
#  [ 4.         5.3132615  6.407606   7.44019  ]
#  [ 8.         9.313262  10.407606  11.440189 ]]

y = paddle.logcumsumexp(data, dtype='float64')
print(y.dtype)
# paddle.float64