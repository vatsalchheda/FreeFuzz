results = dict()
import paddle
import time
arg_1_tensor = paddle.rand([1, 8000], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 512
arg_3 = 458
arg_4 = "circular"
arg_5_tensor = paddle.rand([512], dtype=paddle.float32)
arg_5 = arg_5_tensor.clone()
arg_6 = True
arg_7 = "reflect"
start = time.time()
results["time_low"] = paddle.signal.stft(arg_1,n_fft=arg_2,hop_length=arg_3,win_length=arg_4,window=arg_5,center=arg_6,pad_mode=arg_7,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().astype(paddle.float32)
arg_5 = arg_5_tensor.clone().astype(paddle.float32)
start = time.time()
results["time_high"] = paddle.signal.stft(arg_1,n_fft=arg_2,hop_length=arg_3,win_length=arg_4,window=arg_5,center=arg_6,pad_mode=arg_7,)
results["time_high"] = time.time() - start

print(results)
