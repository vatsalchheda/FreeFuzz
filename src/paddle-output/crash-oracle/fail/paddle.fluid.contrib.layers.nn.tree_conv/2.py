import paddle
arg_1_tensor = paddle.rand([-1, 10, 6], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([-1, 10, 2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = 3
arg_4 = 4
arg_5 = -1024
res = paddle.fluid.contrib.layers.nn.tree_conv(arg_1,arg_2,arg_3,arg_4,arg_5,)
