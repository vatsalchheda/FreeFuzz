import paddle
arg_1_tensor = paddle.rand([-1, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1,1024,[1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.fluid.layers.control_flow.array_read(array=arg_1,i=arg_2,)
