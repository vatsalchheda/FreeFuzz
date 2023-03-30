import paddle

n_fft = 512
cosine_window = paddle.audio.functional.get_window('cosine', n_fft)

std = 7
gaussian_window = paddle.audio.functional.get_window(('gaussian',std), n_fft)