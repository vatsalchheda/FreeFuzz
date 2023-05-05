import paddle
arg_1_tensor = paddle.randint(-64, 2, [11, 4, 4], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8192, 128, [11, 4, 4], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
res = paddle.fluid.layers.nn.gather_tree(arg_1,arg_2,)
