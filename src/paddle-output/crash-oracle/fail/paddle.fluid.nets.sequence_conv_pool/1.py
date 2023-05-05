import paddle
arg_1_tensor = paddle.rand([-1, 128], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 472
arg_3 = 32
arg_4 = "tanh"
arg_5 = "sqrt"
res = paddle.fluid.nets.sequence_conv_pool(input=arg_1,num_filters=arg_2,filter_size=arg_3,act=arg_4,pool_type=arg_5,)
