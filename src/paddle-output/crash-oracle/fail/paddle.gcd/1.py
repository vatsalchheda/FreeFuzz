import paddle
arg_1_tensor = paddle.randint(-512, 1024, [1], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.gcd(arg_1,arg_2,)
