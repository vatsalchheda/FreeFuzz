import paddle
arg_1_tensor = paddle.randint(-8, 512, [1], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4096, 64, [1], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
arg_3 = False
arg_4_tensor = paddle.randint(0,2,[1])
arg_4 = arg_4_tensor.clone()
res = paddle.fluid.layers.control_flow.less_than(x=arg_1,y=arg_2,force_cpu=arg_3,cond=arg_4,)
