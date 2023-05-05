import paddle
arg_1_tensor = paddle.rand([1, 1000216], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 5
arg_3 = 1
res = paddle.topk(arg_1,arg_2,axis=arg_3,)
