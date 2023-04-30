import paddle
arg_1_tensor = paddle.randint(-8,16,[4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8,2,[4], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = 0.5
res = paddle.lerp(arg_1,arg_2,arg_3,)
