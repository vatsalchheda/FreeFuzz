import paddle
arg_1_tensor = paddle.rand([1, 32, 40800], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.1
arg_3 = None
res = paddle.nn.functional.leaky_relu(arg_1,arg_2,arg_3,)
