import paddle
arg_1_tensor = paddle.rand([4, 4], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = 35
res = paddle.fluid.contrib.sparsity.utils.check_mask_1d(arg_1,n=arg_2,m=arg_3,)
