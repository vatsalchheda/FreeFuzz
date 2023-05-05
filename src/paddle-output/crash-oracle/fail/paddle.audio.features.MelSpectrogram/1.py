import paddle
arg_1 = 32000
arg_2 = 1024
arg_3 = 324
arg_4 = 1024
arg_5 = "hann"
arg_6 = 2.0
arg_7 = True
arg_8 = "reflect"
arg_9 = -88.0
arg_10 = 1.0
arg_11 = 14000.0
arg_12 = True
arg_13 = "slaney"
arg_14 = "float32"
arg_class = paddle.audio.features.MelSpectrogram(sr=arg_1,n_fft=arg_2,hop_length=arg_3,win_length=arg_4,window=arg_5,power=arg_6,center=arg_7,pad_mode=arg_8,n_mels=arg_9,f_min=arg_10,f_max=arg_11,htk=arg_12,norm=arg_13,dtype=arg_14,)
arg_15_0_tensor = paddle.rand([1, 159898], dtype=paddle.float32)
arg_15_0 = arg_15_0_tensor.clone()
arg_15 = [arg_15_0,]
res = arg_class(*arg_15)
