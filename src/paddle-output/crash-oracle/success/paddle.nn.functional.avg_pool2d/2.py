import paddle
arg_1_tensor = paddle.rand([1, 64, 500, 64], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 6
res = paddle.nn.functional.avg_pool2d(arg_1,kernel_size=arg_2,)
