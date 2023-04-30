import paddle
arg_1_tensor = paddle.randint(-256,1024,[3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2,2048,[3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = 1
res = paddle.cross(arg_1,arg_2,axis=arg_3,)
