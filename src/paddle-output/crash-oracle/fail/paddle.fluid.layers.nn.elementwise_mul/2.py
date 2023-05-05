import paddle
arg_1_tensor = paddle.rand([4, 4], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([4], dtype=paddle.float64)
arg_2 = arg_2_tensor.clone()
arg_3 = -16
res = paddle.fluid.layers.nn.elementwise_mul(arg_1,arg_2,axis=arg_3,)
