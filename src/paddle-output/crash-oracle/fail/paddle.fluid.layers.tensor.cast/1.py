import paddle
arg_1_tensor = paddle.randint(-16384, 4096, [1], dtype=paddle.int32arg_1 = arg_1_tensor.clone()
arg_2 = "paddleVarType"
res = paddle.fluid.layers.tensor.cast(arg_1,arg_2,)
