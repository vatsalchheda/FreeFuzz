import paddle
arg_1_tensor = paddle.randint(-256,8192,[2, 3, 4, 5, 6], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.nn.functional.dropout3d(arg_1,)
