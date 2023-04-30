import paddle
arg_1_tensor = paddle.randint(-1024,32768,[3], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
res = paddle.fluid.layers.tensor.assign(arg_1,)
