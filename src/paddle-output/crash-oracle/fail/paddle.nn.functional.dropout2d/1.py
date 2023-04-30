import paddle
arg_1_tensor = paddle.randint(-1024,128,[2, 3, 4, 5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.nn.functional.dropout2d(arg_1,)
