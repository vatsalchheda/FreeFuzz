import paddle
arg_1_tensor = paddle.rand([10, 4, 8], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -16
res = paddle.fluid.layers.nn.topk(arg_1,k=arg_2,)
