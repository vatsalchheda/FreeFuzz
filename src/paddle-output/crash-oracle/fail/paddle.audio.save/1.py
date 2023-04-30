import paddle
arg_1 = "circular"
arg_2_tensor = paddle.randint(-2,512,[1, 8000], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = 16000
res = paddle.audio.save(arg_1,arg_2,arg_3,)
