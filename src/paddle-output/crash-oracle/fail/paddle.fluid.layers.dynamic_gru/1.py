import paddle
arg_1_tensor = paddle.rand([-1, 1536], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 574
res = paddle.fluid.layers.dynamic_gru(input=arg_1,size=arg_2,)
