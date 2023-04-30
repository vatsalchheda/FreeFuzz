results = dict()
import paddle
arg_1_tensor = paddle.randint(-128,512,[1, 40190], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = "max"
arg_3_0 = -18
arg_3_1 = 42
arg_3 = (arg_3_0,arg_3_1,)
arg_4 = 512
arg_5_tensor = paddle.randint(-16384,1024,[512], dtype=paddle.float32)
arg_5 = arg_5_tensor.clone()
arg_6 = True
arg_7 = "reflect"
try:
  results["res_cpu"] = paddle.signal.stft(arg_1,n_fft=arg_2,hop_length=arg_3,win_length=arg_4,window=arg_5,center=arg_6,pad_mode=arg_7,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_3 = (arg_3_0,arg_3_1,)
arg_5 = arg_5_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.signal.stft(arg_1,n_fft=arg_2,hop_length=arg_3,win_length=arg_4,window=arg_5,center=arg_6,pad_mode=arg_7,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
