import paddle
arg_1_tensor = paddle.rand([10, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([10, 1], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.nn.functional.log_loss(input=arg_1,label=arg_2,)
