import paddle
arg_1_tensor = paddle.randint(-128,1,[3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-64,256,[3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.cross(arg_1,arg_2,)
