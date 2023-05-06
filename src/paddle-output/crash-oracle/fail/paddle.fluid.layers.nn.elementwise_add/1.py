import paddle
arg_1_tensor = paddle.rand([1, 64, 1, 41100], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([64], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_0 = 1
arg_3_1 = 2
arg_3 = [arg_3_0,arg_3_1,]
res = paddle.fluid.layers.nn.elementwise_add(arg_1,arg_2,axis=arg_3,)
