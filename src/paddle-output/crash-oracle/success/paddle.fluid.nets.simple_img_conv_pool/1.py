import paddle
arg_1_tensor = paddle.rand([57, 1, 28, 1024], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 3
arg_3 = 20
arg_4 = 2
arg_5 = 2
arg_6 = "relu"
res = paddle.fluid.nets.simple_img_conv_pool(input=arg_1,filter_size=arg_2,num_filters=arg_3,pool_size=arg_4,pool_stride=arg_5,act=arg_6,)
