import paddle

paddle.device.set_device("cpu")
x1 = paddle.ones(name='x1', shape=[1, 2], dtype='int32')
x2 = paddle.zeros(name='x2', shape=[1, 2], dtype='int32')
data = paddle.stack([x1,x2], axis=1)