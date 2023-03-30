import paddle

n_mels = 64
f_min = 0.5
f_max = 10000
htk_flag = True

paddle_mel_freq = paddle.audio.functional.mel_frequencies(
    n_mels, f_min, f_max, htk_flag, 'float64')