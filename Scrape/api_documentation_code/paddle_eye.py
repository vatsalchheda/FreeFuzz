import paddle

data = paddle.eye(3, dtype='int32')
# [[1 0 0]
#  [0 1 0]
#  [0 0 1]]
data = paddle.eye(2, 3, dtype='int32')
# [[1 0 0]
#  [0 1 0]]