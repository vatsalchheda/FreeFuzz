import paddle
arg_1_tensor = paddle.rand([4, 4], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = 4
res = paddle.fluid.contrib.sparsity.utils.get_mask_2d_best(arg_1,n=arg_2,m=arg_3,)
