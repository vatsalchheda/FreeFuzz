import paddle
arg_1 = None
arg_2 = -13.76393202250021
arg_3 = "leaky_relu"
arg_class = paddle.nn.initializer.KaimingUniform(fan_in=arg_1,negative_slope=arg_2,nonlinearity=arg_3,)
arg_4_0_tensor = paddle.rand([2048], dtype=paddle.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4_1_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_4_1 = arg_4_1_tensor.clone()
arg_4 = [arg_4_0,arg_4_1,]
res = arg_class(*arg_4)
