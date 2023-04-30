import paddle
arg_1_tensor = paddle.randint(-32,4,[4, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1024,8,[3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.inner(arg_1,arg_2,)
