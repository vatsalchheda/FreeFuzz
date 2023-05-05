import paddle
arg_1_tensor = paddle.randint(0,2,[1, 1])
arg_1 = arg_1_tensor.clone()
arg_2 = 501
res = paddle.signal.istft(arg_1,n_fft=arg_2,)
