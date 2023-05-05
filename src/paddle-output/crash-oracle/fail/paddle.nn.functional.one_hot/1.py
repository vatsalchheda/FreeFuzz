import paddle
arg_1_tensor = paddle.randint(-16384, 2048, [10, 2, 3], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2 = -1
res = paddle.nn.functional.one_hot(arg_1,arg_2,)
