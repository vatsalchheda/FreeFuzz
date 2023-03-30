import paddle

sr = 16000
n_fft = 128
fft_freq = paddle.audio.functional.fft_frequencies(sr, n_fft)