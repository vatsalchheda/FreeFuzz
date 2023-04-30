import paddle
arg_1_tensor = paddle.randint(-8,2,[4, 45], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.is_floating_point(arg_1,)
