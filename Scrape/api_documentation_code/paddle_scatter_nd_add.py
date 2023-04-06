import paddle

x = paddle.rand(shape=[3, 5, 9, 10], dtype='float32')
updates = paddle.rand(shape=[3, 9, 10], dtype='float32')
index = paddle.to_tensor([[1, 1],
                        [0, 1],
                        [1, 3]], dtype='int64')

output = paddle.scatter_nd_add(x, index, updates)
print(output.shape)
# [3, 5, 9, 10]