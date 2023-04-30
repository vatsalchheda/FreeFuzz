import paddle
arg_1_tensor = paddle.randint(-8,4,[4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4096,32768,[5], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.floor_divide(arg_1,arg_2,)
