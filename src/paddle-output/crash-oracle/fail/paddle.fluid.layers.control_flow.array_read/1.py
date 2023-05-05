import paddle
arg_1_tensor = paddle.rand([-1, 128], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1024, 2048, [1], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
res = paddle.fluid.layers.control_flow.array_read(arg_1,arg_2,)
