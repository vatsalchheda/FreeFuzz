import paddle
arg_1_tensor = paddle.randint(-4, 8, [3, 224, 224], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
res = paddle.nn.functional.one_hot(arg_1,arg_2,)
