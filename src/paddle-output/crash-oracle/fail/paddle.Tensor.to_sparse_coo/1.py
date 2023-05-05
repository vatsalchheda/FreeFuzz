import paddle
arg_1_tensor = paddle.rand([3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -8
res = paddle.Tensor.to_sparse_coo(arg_1,arg_2,)
