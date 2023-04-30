import paddle
arg_1_tensor = paddle.randint(-8,2048,[38], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = "sum"
res = paddle.fluid.layers.tensor.cast(x=arg_1,dtype=arg_2,)
