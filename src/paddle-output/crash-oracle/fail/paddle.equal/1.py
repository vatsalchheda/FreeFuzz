import paddle
arg_1_tensor = paddle.randint(-256, 1, [6], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2 = 0
res = paddle.equal(arg_1,arg_2,)
