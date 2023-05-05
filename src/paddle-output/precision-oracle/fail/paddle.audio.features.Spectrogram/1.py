results = dict()
import paddle
import time
arg_1 = 512
arg_2 = None
arg_3 = None
arg_4 = "hann"
arg_5 = 1024.0
arg_6 = -79.0
arg_7 = "reflect"
arg_8 = "float32"
arg_class = paddle.audio.features.Spectrogram(n_fft=arg_1,hop_length=arg_2,win_length=arg_3,window=arg_4,power=arg_5,center=arg_6,pad_mode=arg_7,dtype=arg_8,)
float_tensor = paddle.rand([1, 220500], 'float32')
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
