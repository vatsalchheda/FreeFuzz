import paddle
arg_1 = -60.17677669529664
arg_2 = 43.17677669529664
arg_3 = 0
arg_4 = 0
arg_5 = 103.0
arg_6 = -1024.0
arg_class = paddle.fluid.initializer.UniformInitializer(low=arg_1,high=arg_2,seed=arg_3,diag_num=arg_4,diag_step=arg_5,diag_val=arg_6,)
arg_7_0_tensor = paddle.rand([128], dtype=paddle.float32)
arg_7_0 = arg_7_0_tensor.clone()
arg_7_1_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_7_1 = arg_7_1_tensor.clone()
arg_7 = [arg_7_0,arg_7_1,]
res = arg_class(*arg_7)
