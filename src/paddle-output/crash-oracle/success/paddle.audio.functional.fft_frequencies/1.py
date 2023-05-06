import paddle
arg_1 = 16000
arg_2 = 128
res = paddle.audio.functional.fft_frequencies(arg_1,arg_2,)
