import paddle
arg_1_tensor = paddle.rand([4, 4, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([32], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = -20
res = paddle.fluid.layers.nn.elementwise_mul(arg_1,arg_2,axis=arg_3,)
