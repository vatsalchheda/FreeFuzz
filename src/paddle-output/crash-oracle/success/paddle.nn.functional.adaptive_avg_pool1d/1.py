import paddle
arg_1_tensor = paddle.rand([1, 512, 136], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 72
arg_3 = None
res = paddle.nn.functional.adaptive_avg_pool1d(arg_1,arg_2,arg_3,)
