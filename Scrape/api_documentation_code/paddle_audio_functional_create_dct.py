import paddle
n_mfcc = 23
n_mels = 257
dct = paddle.audio.functional.create_dct(n_mfcc, n_mels)