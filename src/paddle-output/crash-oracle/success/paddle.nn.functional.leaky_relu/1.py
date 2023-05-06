import paddle
arg_1_tensor = paddle.rand([1, 64, 13600], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -48
arg_3 = None
res = paddle.nn.functional.leaky_relu(arg_1,arg_2,arg_3,)
