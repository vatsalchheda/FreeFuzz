import paddle
arg_1 = 563
arg_2 = 1
arg_3 = None
arg_4 = "hann"
arg_5 = 1.0
arg_6 = True
arg_7 = 63.0
arg_8 = "max"
arg_class = paddle.audio.features.Spectrogram(n_fft=arg_1,hop_length=arg_2,win_length=arg_3,window=arg_4,power=arg_5,center=arg_6,pad_mode=arg_7,dtype=arg_8,)
arg_9_0_tensor = paddle.rand([1, 8000], dtype=paddle.float32)
arg_9_0 = arg_9_0_tensor.clone()
arg_9 = [arg_9_0,]
res = arg_class(*arg_9)
