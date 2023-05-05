results = dict()
import paddle
import time
arg_1 = 92.0
arg_2 = None
arg_3 = None
arg_4 = "max"
arg_5 = 52.0
arg_6 = True
arg_7 = "mean"
arg_8 = "float32"
arg_class = paddle.audio.features.Spectrogram(n_fft=arg_1,hop_length=arg_2,win_length=arg_3,window=arg_4,power=arg_5,center=arg_6,pad_mode=arg_7,dtype=arg_8,)
arg_9_0_tensor = paddle.rand([1, 40190], dtype=paddle.float32)
arg_9_0 = arg_9_0_tensor.clone()
arg_9 = [arg_9_0,]
start = time.time()
results["time_low"] = arg_class(*arg_9)
results["time_low"] = time.time() - start
arg_9_0 = arg_9_0_tensor.clone().astype(paddle.float32)
arg_9 = [arg_9_0,]
start = time.time()
results["time_high"] = arg_class(*arg_9)
results["time_high"] = time.time() - start

print(results)
