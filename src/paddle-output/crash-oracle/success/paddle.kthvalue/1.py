import paddle
arg_1_tensor = paddle.rand([2, 3, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = 1
res = paddle.kthvalue(arg_1,arg_2,arg_3,)
