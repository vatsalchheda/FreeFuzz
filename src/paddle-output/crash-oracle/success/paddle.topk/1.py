import paddle
arg_1_tensor = paddle.rand([1, 30001], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 5
res = paddle.topk(arg_1,k=arg_2,)
