import paddle
arg_1 = 512
arg_2 = None
arg_3 = None
arg_4 = "hann"
arg_5 = 1024.0
arg_6 = -79.0
arg_7 = "reflect"
arg_8 = "float32"
arg_class = paddle.audio.features.Spectrogram(n_fft=arg_1,hop_length=arg_2,win_length=arg_3,window=arg_4,power=arg_5,center=arg_6,pad_mode=arg_7,dtype=arg_8,)
arg_9_0_tensor = paddle.rand([1, 220500], dtype=paddle.float32)
arg_9_0 = arg_9_0_tensor.clone()
arg_9 = [arg_9_0,]
res = arg_class(*arg_9)
