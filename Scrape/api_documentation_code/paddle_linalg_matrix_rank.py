import paddle

a = paddle.eye(10)
b = paddle.linalg.matrix_rank(a)
print(b)
# b = [10]

c = paddle.ones(shape=[3, 4, 5, 5])
d = paddle.linalg.matrix_rank(c, tol=0.01, hermitian=True)
print(d)
# d = [[1, 1, 1, 1],
#      [1, 1, 1, 1],
#      [1, 1, 1, 1]]