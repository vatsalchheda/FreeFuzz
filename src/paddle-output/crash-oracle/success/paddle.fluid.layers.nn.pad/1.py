import paddle
arg_1_tensor = paddle.rand([-1, -1, 128], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 0
arg_2_1 = 0
arg_2_2 = 0
arg_2_3 = 1
arg_2_4 = 0
arg_2_5 = 0
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,arg_2_4,arg_2_5,]
res = paddle.fluid.layers.nn.pad(arg_1,paddings=arg_2,)
