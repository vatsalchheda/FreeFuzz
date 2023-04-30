import paddle
arg_1_tensor = paddle.randint(-2,8192,[5], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 3
res = paddle.nn.functional.pixel_shuffle(arg_1,arg_2,)
