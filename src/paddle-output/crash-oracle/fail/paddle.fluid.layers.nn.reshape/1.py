import paddle
arg_1_tensor = paddle.rand([64, 64, 7, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 16
arg_2_1 = 30
arg_2_2 = 61
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
arg_3 = "Normal_sample"
res = paddle.fluid.layers.nn.reshape(arg_1,arg_2,name=arg_3,)
