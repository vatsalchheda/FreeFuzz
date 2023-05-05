import paddle
arg_1_tensor = paddle.randint(-4,32768,[1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1024,32768,[1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.gcd(arg_1,arg_2,)
