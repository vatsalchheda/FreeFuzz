import paddle
arg_1_tensor = paddle.randint(-8,8192,[1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = -29
res = paddle.equal(arg_1,arg_2,)
