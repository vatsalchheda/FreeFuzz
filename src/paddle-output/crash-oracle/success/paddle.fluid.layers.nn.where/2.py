import paddle
arg_1_tensor = paddle.randint(0,2,[2, 16])
arg_1 = arg_1_tensor.clone()
res = paddle.fluid.layers.nn.where(arg_1,)
