import paddle
arg_1_tensor = paddle.randint(-4096, 8, [1], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1.0
arg_3 = True
res = paddle.fluid.layers.control_flow.increment(x=arg_1,value=arg_2,in_place=arg_3,)
