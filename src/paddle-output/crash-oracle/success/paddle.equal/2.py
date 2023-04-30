import paddle
arg_1_tensor = paddle.randint(-64,2,[1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = -1024
res = paddle.equal(arg_1,arg_2,)
