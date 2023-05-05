import paddle
arg_1_tensor = paddle.rand([-1, 784], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1024, 4, [-1, 1], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
res = paddle.nn.functional.softmax_with_cross_entropy(arg_1,arg_2,)
