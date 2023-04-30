import paddle
arg_1_tensor = paddle.randint(-128,64,[4, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4,2048,[1024], dtype=paddle.int16)
arg_2 = arg_2_tensor.clone()
res = paddle.Tensor.fill_diagonal_tensor_(arg_1,arg_2,)
