import paddle

shape = paddle.broadcast_shape([2, 1, 3], [1, 3, 1])
# [2, 3, 3]

# shape = paddle.broadcast_shape([2, 1, 3], [3, 3, 1])
# ValueError (terminated with error message).