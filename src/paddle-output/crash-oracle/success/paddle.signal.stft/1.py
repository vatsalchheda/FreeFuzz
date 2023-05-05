import paddle
arg_1_tensor = paddle.rand([8, 48000], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = 512
arg_3 = False
res = paddle.signal.stft(arg_1,n_fft=arg_2,onesided=arg_3,)
