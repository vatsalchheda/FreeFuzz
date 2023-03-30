import paddle

# vector * vector
x = paddle.rand([10])
y = paddle.rand([10])
z = paddle.matmul(x, y)
print(z.shape)
# [1]

# matrix * vector
x = paddle.rand([10, 5])
y = paddle.rand([5])
z = paddle.matmul(x, y)
print(z.shape)
# [10]

# batched matrix * broadcasted vector
x = paddle.rand([10, 5, 2])
y = paddle.rand([2])
z = paddle.matmul(x, y)
print(z.shape)
# [10, 5]

# batched matrix * batched matrix
x = paddle.rand([10, 5, 2])
y = paddle.rand([10, 2, 5])
z = paddle.matmul(x, y)
print(z.shape)
# [10, 5, 5]

# batched matrix * broadcasted matrix
x = paddle.rand([10, 1, 5, 2])
y = paddle.rand([1, 3, 2, 5])
z = paddle.matmul(x, y)
print(z.shape)
# [10, 3, 5, 5]