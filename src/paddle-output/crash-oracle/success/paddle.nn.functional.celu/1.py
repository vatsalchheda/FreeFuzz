import paddle
arg_1_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.2
res = paddle.nn.functional.celu(arg_1,alpha=arg_2,)
