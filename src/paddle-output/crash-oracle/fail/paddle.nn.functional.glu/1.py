import paddle
arg_1_tensor = paddle.rand([1, 1024, 123], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -31
res = paddle.nn.functional.glu(arg_1,axis=arg_2,)
