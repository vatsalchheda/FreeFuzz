import paddle
arg_1_tensor = paddle.rand([-1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32768, 128, [1], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([-1, 128], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
res = paddle.fluid.layers.control_flow.array_write(arg_1,i=arg_2,array=arg_3,)
