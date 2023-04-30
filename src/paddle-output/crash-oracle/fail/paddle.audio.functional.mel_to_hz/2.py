import paddle
arg_1_tensor = paddle.randint(-1,128,[1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = True
res = paddle.audio.functional.mel_to_hz(arg_1,arg_2,)
