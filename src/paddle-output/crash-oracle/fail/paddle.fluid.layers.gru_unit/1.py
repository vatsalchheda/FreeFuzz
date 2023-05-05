import paddle
arg_1_tensor = paddle.rand([-1, 30], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([-1, 10], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = -10
res = paddle.fluid.layers.gru_unit(input=arg_1,hidden=arg_2,size=arg_3,)
