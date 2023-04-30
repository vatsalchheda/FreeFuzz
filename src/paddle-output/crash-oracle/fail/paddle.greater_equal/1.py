import paddle
arg_1_tensor = paddle.randint(-2048,8,[2, 2, 1, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-128,4096,[1], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.greater_equal(arg_1,arg_2,)
