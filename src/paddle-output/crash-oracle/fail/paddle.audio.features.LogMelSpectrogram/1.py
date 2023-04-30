import paddle
arg_1 = False
arg_2 = 512
arg_3 = None
arg_4 = None
arg_5 = "hann"
arg_6 = 2.0
arg_7 = True
arg_8 = 56
arg_9 = 64
arg_10 = 50.0
arg_11 = None
arg_12 = False
arg_13 = "sum"
arg_14 = 1.0
arg_15 = 3.0000000001
arg_16 = None
arg_17 = "float32"
arg_class = paddle.audio.features.LogMelSpectrogram(sr=arg_1,n_fft=arg_2,hop_length=arg_3,win_length=arg_4,window=arg_5,power=arg_6,center=arg_7,pad_mode=arg_8,n_mels=arg_9,f_min=arg_10,f_max=arg_11,htk=arg_12,norm=arg_13,ref_value=arg_14,amin=arg_15,top_db=arg_16,dtype=arg_17,)
arg_18_0_tensor = paddle.randint(-16,4096,[1, 35005], dtype=paddle.float32)
arg_18_0 = arg_18_0_tensor.clone()
arg_18 = [arg_18_0,]
res = arg_class(*arg_18)
