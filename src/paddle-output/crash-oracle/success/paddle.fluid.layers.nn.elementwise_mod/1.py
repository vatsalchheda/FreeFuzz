import paddle
arg_1_tensor = paddle.randint(-4096, 2048, [4, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-256, 8192, [1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.fluid.layers.nn.elementwise_mod(arg_1,arg_2,)
