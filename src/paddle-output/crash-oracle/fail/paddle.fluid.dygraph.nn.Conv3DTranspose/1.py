import paddle
arg_1 = True
arg_2 = 52
arg_3 = 3
arg_4 = False
arg_class = paddle.fluid.dygraph.nn.Conv3DTranspose(num_channels=arg_1,num_filters=arg_2,filter_size=arg_3,use_cudnn=arg_4,)
arg_5_0_tensor = paddle.rand([5, 3, 12, 32, 32], dtype=paddle.float32)
arg_5_0 = arg_5_0_tensor.clone()
arg_5 = [arg_5_0,]
res = arg_class(*arg_5)
