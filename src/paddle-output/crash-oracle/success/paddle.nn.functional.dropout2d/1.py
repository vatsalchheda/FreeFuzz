import paddle
arg_1_tensor = paddle.rand([2, 3, 4, 5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 39.0
res = paddle.nn.functional.dropout2d(arg_1,training=arg_2,)
