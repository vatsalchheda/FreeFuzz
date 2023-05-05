import paddle
arg_1_tensor = paddle.rand([1, 512, 5538], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 40
res = paddle.nn.functional.log_softmax(arg_1,axis=arg_2,)
