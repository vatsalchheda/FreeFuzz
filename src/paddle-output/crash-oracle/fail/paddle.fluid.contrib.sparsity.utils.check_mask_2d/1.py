import paddle
arg_1_tensor = paddle.rand([4, 4], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = -2
res = paddle.fluid.contrib.sparsity.utils.check_mask_2d(arg_1,n=arg_2,m=arg_3,)
