import paddle
paddle.enable_static()
W = paddle.static.create_parameter(shape=[784, 200], dtype='float32')