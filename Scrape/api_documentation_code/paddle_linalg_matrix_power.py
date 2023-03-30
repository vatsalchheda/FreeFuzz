import paddle

x = paddle.to_tensor([[1, 2, 3],
                      [1, 4, 9],
                      [1, 8, 27]], dtype='float64')
print(paddle.linalg.matrix_power(x, 2))
# [[6.  , 34. , 102.],
#  [14. , 90. , 282.],
#  [36. , 250., 804.]]

print(paddle.linalg.matrix_power(x, 0))
# [[1., 0., 0.],
#  [0., 1., 0.],
#  [0., 0., 1.]]

print(paddle.linalg.matrix_power(x, -2))
# [[ 12.91666667, -12.75000000,  2.83333333 ],
#  [-7.66666667 ,  8.         , -1.83333333 ],
#  [ 1.80555556 , -1.91666667 ,  0.44444444 ]]