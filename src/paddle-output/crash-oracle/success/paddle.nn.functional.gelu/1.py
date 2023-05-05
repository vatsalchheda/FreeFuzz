import paddle
arg_1_tensor = paddle.rand([1, 164, 4096], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.nn.functional.gelu(arg_1,)
