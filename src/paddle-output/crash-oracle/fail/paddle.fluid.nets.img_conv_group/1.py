import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
arg_3_0 = 43
arg_3_1 = -57
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = 55
arg_5 = "zeros"
arg_6 = 2
arg_7 = 2
res = paddle.fluid.nets.img_conv_group(input=arg_1,conv_padding=arg_2,conv_num_filter=arg_3,conv_filter_size=arg_4,conv_act=arg_5,pool_size=arg_6,pool_stride=arg_7,)
