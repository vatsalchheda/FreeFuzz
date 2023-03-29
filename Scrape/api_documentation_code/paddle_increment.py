import paddle

data = paddle.zeros(shape=[1], dtype='float32')
counter = paddle.increment(data)
# [1.]