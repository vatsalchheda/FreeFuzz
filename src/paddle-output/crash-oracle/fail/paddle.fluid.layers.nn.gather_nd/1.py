import paddle
arg_1_tensor = paddle.randint(0,2,[4, 4])
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-64, 32768, [4, 4, 2], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
res = paddle.fluid.layers.nn.gather_nd(arg_1,arg_2,)
