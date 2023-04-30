import paddle
arg_1_tensor = paddle.randint(-1,8192,[4, 4], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = 4
res = paddle.fluid.contrib.sparsity.utils.get_mask_1d(arg_1,n=arg_2,m=arg_3,)
