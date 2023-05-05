import paddle
arg_1_tensor = paddle.randint(-4096,4096,[3], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-16,128,[2, 3], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
res = paddle.expand_as(arg_1,arg_2,)
