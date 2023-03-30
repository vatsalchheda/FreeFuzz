import paddle

index = paddle.to_tensor([[1, 1],
                        [0, 1],
                        [1, 3]], dtype="int64")
updates = paddle.rand(shape=[3, 9, 10], dtype='float32')
shape = [3, 5, 9, 10]

output = paddle.scatter_nd(index, updates, shape)