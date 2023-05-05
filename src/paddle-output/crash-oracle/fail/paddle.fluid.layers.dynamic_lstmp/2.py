import paddle
arg_1_tensor = paddle.rand([-1, 2048], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 2048
arg_3 = 256
arg_4 = False
arg_5 = True
arg_6 = "tanh"
arg_7 = "tanh"
res = paddle.fluid.layers.dynamic_lstmp(input=arg_1,size=arg_2,proj_size=arg_3,use_peepholes=arg_4,is_reverse=arg_5,cell_activation=arg_6,proj_activation=arg_7,)
