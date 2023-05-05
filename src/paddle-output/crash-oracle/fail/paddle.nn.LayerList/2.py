import paddle
arg_class = paddle.nn.LayerList()
arg_1_0_tensor = paddle.rand([4, 23, 16], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.rand([2, 4, 32], dtype=paddle.float32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
res = arg_class(*arg_1)
