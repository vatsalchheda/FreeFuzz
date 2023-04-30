import paddle
arg_1 = "./test.wav"
arg_2_tensor = paddle.randint(-8192,2048,[1, 8000], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = 15971
res = paddle.audio.save(arg_1,arg_2,arg_3,)
