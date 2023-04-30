import paddle
arg_1_tensor = paddle.randint(-2048,1024,[3, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32,32768,[3, 2], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-128,32768,[53, 9], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
res = paddle.scatter_nd_add(arg_1,arg_2,arg_3,)
