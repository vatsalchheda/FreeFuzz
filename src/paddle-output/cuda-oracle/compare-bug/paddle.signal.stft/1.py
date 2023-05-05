results = dict()
import paddle
arg_1_tensor = paddle.rand([8, 48000], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = 512
arg_3 = False
try:
  results["res_cpu"] = paddle.signal.stft(arg_1,n_fft=arg_2,onesided=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.signal.stft(arg_1,n_fft=arg_2,onesided=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
