import paddle
arg_1_tensor = paddle.rand([1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.audio.functional.power_to_db(arg_1,)
