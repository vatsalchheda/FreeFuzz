import paddle
arg_1_tensor = paddle.rand([1, 220500], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 490
arg_3 = None
arg_4 = 512
arg_5_tensor = paddle.rand([512], dtype=paddle.float32)
arg_5 = arg_5_tensor.clone()
arg_6 = False
arg_7 = "reflect"
res = paddle.signal.stft(arg_1,n_fft=arg_2,hop_length=arg_3,win_length=arg_4,window=arg_5,center=arg_6,pad_mode=arg_7,)
