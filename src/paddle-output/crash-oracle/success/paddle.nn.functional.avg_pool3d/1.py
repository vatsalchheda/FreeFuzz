import paddle
arg_1_tensor = paddle.rand([3, 1, 7, 112, 112], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 5
arg_2_1 = 1
arg_2_2 = 1
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
arg_3 = 40
res = paddle.nn.functional.avg_pool3d(arg_1,kernel_size=arg_2,stride=arg_3,)
