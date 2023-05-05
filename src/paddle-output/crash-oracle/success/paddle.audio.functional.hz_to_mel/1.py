import paddle
arg_1_tensor = paddle.rand([1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = False
res = paddle.audio.functional.hz_to_mel(arg_1,arg_2,)
