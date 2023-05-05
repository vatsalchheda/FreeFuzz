import paddle
arg_1_tensor = paddle.rand([32, 32, 7, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 32
arg_2_1 = 32
arg_2_2 = 7
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
arg_3 = -92
res = paddle.fluid.layers.nn.reshape(arg_1,arg_2,name=arg_3,)
