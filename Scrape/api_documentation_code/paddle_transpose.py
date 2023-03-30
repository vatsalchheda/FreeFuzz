import paddle

x = paddle.randn([2, 3, 4])
x_transposed = paddle.transpose(x, perm=[1, 0, 2])
print(x_transposed.shape)
# [3L, 2L, 4L]