import paddle
arg_1_tensor = paddle.rand([4, 2, 1, 71], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.nn.functional.softmax(arg_1,)
