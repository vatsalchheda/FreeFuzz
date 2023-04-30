import paddle
arg_1_tensor = paddle.randint(-16384,1024,[3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32768,4096,[2], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
arg_3 = 1
arg_4_tensor = paddle.randint(-64,128,[3, 2], dtype=paddle.float32)
arg_4 = arg_4_tensor.clone()
res = paddle.index_add_(arg_1,arg_2,arg_3,arg_4,)
