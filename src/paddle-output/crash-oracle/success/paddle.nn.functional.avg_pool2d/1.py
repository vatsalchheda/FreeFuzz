import paddle
arg_1_tensor = paddle.rand([1, 256, 126, 16], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 2
arg_2_1 = 2
arg_2 = [arg_2_0,arg_2_1,]
res = paddle.nn.functional.avg_pool2d(arg_1,kernel_size=arg_2,)
