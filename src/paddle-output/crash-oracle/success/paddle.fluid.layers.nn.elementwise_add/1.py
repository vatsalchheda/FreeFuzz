import paddle
arg_1_tensor = paddle.rand([1, 128, 1, 709], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([128], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = 1
res = paddle.fluid.layers.nn.elementwise_add(arg_1,arg_2,axis=arg_3,)
