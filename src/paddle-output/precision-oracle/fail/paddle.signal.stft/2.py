results = dict()
import paddle
import time
float_tensor = paddle.rand([1, 220500], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
arg_2 = 490
arg_3 = None
arg_4 = 512
float_tensor = paddle.rand([512], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_5_tensor = f16_tensor
arg_5 = arg_5_tensor.clone()
arg_6 = False
arg_7 = "reflect"
start = time.time()
results["time_low"] = paddle.signal.stft(arg_1,n_fft=arg_2,hop_length=arg_3,win_length=arg_4,window=arg_5,center=arg_6,pad_mode=arg_7,)
results["time_low"] = time.time() - start
arg_1 = arg_1_tensor.clone().type(paddle.float32)
arg_5 = arg_5_tensor.clone().type(paddle.float32)
start = time.time()
results["time_high"] = paddle.signal.stft(arg_1,n_fft=arg_2,hop_length=arg_3,win_length=arg_4,window=arg_5,center=arg_6,pad_mode=arg_7,)
results["time_high"] = time.time() - start

print(results)
