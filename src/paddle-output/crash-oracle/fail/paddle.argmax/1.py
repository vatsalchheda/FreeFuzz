import paddle
arg_1_tensor = paddle.rand([2, 3, 32, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -33
res = paddle.argmax(arg_1,axis=arg_2,)
