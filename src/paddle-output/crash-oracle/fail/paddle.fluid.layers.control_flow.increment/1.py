import paddle
arg_1_tensor = paddle.randint(-32768, 8192, [1], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2 = 1.0
arg_3 = True
res = paddle.fluid.layers.control_flow.increment(x=arg_1,value=arg_2,in_place=arg_3,)
