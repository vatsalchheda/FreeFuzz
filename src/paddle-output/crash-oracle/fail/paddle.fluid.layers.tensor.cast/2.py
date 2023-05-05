import paddle
arg_1_tensor = paddle.randint(0,2,[4, 4])
arg_1 = arg_1_tensor.clone()
arg_2 = "zeros"
res = paddle.fluid.layers.tensor.cast(arg_1,arg_2,)
