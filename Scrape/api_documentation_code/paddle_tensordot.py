import paddle

data_type = 'float64'

# For two 2-d tensor x and y, the case axes=0 is equivalent to outer product.
# Note that tensordot supports empty axis sequence, so all the axes=0, axes=[], axes=[[]], and axes=[[],[]] are equivalent cases.
x = paddle.arange(4, dtype=data_type).reshape([2, 2])
y = paddle.arange(4, dtype=data_type).reshape([2, 2])
z = paddle.tensordot(x, y, axes=0)
# z = [[[[0., 0.],
#        [0., 0.]],
#
#       [[0., 1.],
#        [2., 3.]]],
#
#
#      [[[0., 2.],
#        [4., 6.]],
#
#       [[0., 3.],
#        [6., 9.]]]]


# For two 1-d tensor x and y, the case axes=1 is equivalent to inner product.
x = paddle.arange(10, dtype=data_type)
y = paddle.arange(10, dtype=data_type)
z1 = paddle.tensordot(x, y, axes=1)
z2 = paddle.dot(x, y)
# z1 = z2 = [285.]


# For two 2-d tensor x and y, the case axes=1 is equivalent to matrix multiplication.
x = paddle.arange(6, dtype=data_type).reshape([2, 3])
y = paddle.arange(12, dtype=data_type).reshape([3, 4])
z1 = paddle.tensordot(x, y, axes=1)
z2 = paddle.matmul(x, y)
# z1 = z2 =  [[20., 23., 26., 29.],
#             [56., 68., 80., 92.]]


# When axes is a 1-d int list, x and y will be contracted along the same given axes.
# Note that axes=[1, 2] is equivalent to axes=[[1, 2]], axes=[[1, 2], []], axes=[[1, 2], [1]], and axes=[[1, 2], [1, 2]].
x = paddle.arange(24, dtype=data_type).reshape([2, 3, 4])
y = paddle.arange(36, dtype=data_type).reshape([3, 3, 4])
z = paddle.tensordot(x, y, axes=[1, 2])
# z =  [[506. , 1298., 2090.],
#       [1298., 3818., 6338.]]


# When axes is a list containing two 1-d int list, the first will be applied to x and the second to y.
x = paddle.arange(60, dtype=data_type).reshape([3, 4, 5])
y = paddle.arange(24, dtype=data_type).reshape([4, 3, 2])
z = paddle.tensordot(x, y, axes=([1, 0], [0, 1]))
# z =  [[4400., 4730.],
#       [4532., 4874.],
#       [4664., 5018.],
#       [4796., 5162.],
#       [4928., 5306.]]


# Thanks to the support of axes expansion, axes=[[0, 1, 3, 4], [1, 0, 3, 4]] can be abbreviated as axes= [[0, 1, 3, 4], [1, 0]].
x = paddle.arange(720, dtype=data_type).reshape([2, 3, 4, 5, 6])
y = paddle.arange(720, dtype=data_type).reshape([3, 2, 4, 5, 6])
z = paddle.tensordot(x, y, axes=[[0, 1, 3, 4], [1, 0]])
# z = [[23217330., 24915630., 26613930., 28312230.],
#      [24915630., 26775930., 28636230., 30496530.],
#      [26613930., 28636230., 30658530., 32680830.],
#      [28312230., 30496530., 32680830., 34865130.]]