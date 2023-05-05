import paddle
arg_1_tensor = paddle.rand([1, 1, 1, 11], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1024,4096,[4], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.gather(arg_1,arg_2,)
