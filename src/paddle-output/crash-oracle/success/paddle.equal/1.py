import paddle
arg_1_tensor = paddle.randint(-1024, 256, [1], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8, 8, [1], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
res = paddle.equal(arg_1,arg_2,)
