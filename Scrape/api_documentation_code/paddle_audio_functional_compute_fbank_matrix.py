import paddle

n_mfcc = 23
n_mels = 51
paddle_dct = paddle.audio.functional.create_dct(n_mfcc, n_mels)