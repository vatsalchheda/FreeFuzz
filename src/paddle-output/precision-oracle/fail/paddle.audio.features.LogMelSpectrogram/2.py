results = dict()
import paddle
import time
arg_1 = 32000
arg_2 = 1010
arg_3 = 320
arg_4 = -47.0
arg_5 = 1024
arg_6 = False
arg_7 = 13976.0
arg_8 = 64
arg_class = paddle.audio.features.LogMelSpectrogram(sr=arg_1,n_fft=arg_2,hop_length=arg_3,window=arg_4,win_length=arg_5,f_min=arg_6,f_max=arg_7,n_mels=arg_8,)
float_tensor = paddle.rand([1, 162168], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_9_0_tensor = f16_tensor
arg_9_0 = arg_9_0_tensor.clone()
arg_9 = [arg_9_0,]
start = time.time()
results["time_low"] = arg_class(*arg_9)
results["time_low"] = time.time() - start
arg_9_0 = arg_9_0_tensor.clone().type(paddle.float32)
arg_9 = [arg_9_0,]
start = time.time()
results["time_high"] = arg_class(*arg_9)
results["time_high"] = time.time() - start

print(results)
