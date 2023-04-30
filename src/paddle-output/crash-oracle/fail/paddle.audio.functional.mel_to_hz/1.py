import paddle
arg_1_tensor = paddle.randint(-4,2,[1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = False
res = paddle.audio.functional.mel_to_hz(arg_1,arg_2,)
