import paddle
arg_1 = 31
arg_class = paddle.fluid.dygraph.nn.Dropout(p=arg_1,)
arg_2_0_tensor = paddle.rand([3, 10, 0, 7, 61], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
res = arg_class(*arg_2)
