import paddle
arg_1_tensor = paddle.randint(-128, 32, [4, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1, 32, [4, 4, 2], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.fluid.layers.nn.gather_nd(arg_1,arg_2,)
