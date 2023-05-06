results = dict()
import paddle
import time
arg_1 = 28.0
arg_2 = 474
arg_3 = None
arg_4 = None
arg_5 = "hann"
arg_6 = 2.0
arg_7 = 69.0
arg_8 = "reflect"
arg_9 = 64
arg_10 = 16.0
arg_11 = None
arg_12 = True
arg_13 = 117.0
arg_14 = -23.0
arg_15 = 1e-10
arg_16 = None
arg_17 = "float32"
arg_class = paddle.audio.features.LogMelSpectrogram(sr=arg_1,n_fft=arg_2,hop_length=arg_3,win_length=arg_4,window=arg_5,power=arg_6,center=arg_7,pad_mode=arg_8,n_mels=arg_9,f_min=arg_10,f_max=arg_11,htk=arg_12,norm=arg_13,ref_value=arg_14,amin=arg_15,top_db=arg_16,dtype=arg_17,)
float_tensor = paddle.rand([1, 220500], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_18_0_tensor = f16_tensor
arg_18_0 = arg_18_0_tensor.clone()
arg_18 = [arg_18_0,]
start = time.time()
results["time_low"] = arg_class(*arg_18)
results["time_low"] = time.time() - start
arg_18_0 = arg_18_0_tensor.clone().type(paddle.float32)
arg_18 = [arg_18_0,]
start = time.time()
results["time_high"] = arg_class(*arg_18)
results["time_high"] = time.time() - start

print(results)
