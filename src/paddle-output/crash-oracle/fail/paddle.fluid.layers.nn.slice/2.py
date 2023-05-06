import paddle
arg_1_tensor = paddle.rand([-1, -1, 128], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2 = [arg_2_0,]
arg_3_0_tensor = paddle.randint(-128, 32, [], dtype=paddle.int32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
arg_4_0_tensor = paddle.randint(-4, 32768, [1], dtype=paddle.int32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
res = paddle.fluid.layers.nn.slice(arg_1,axes=arg_2,starts=arg_3,ends=arg_4,)
