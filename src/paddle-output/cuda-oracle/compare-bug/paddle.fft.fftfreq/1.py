results = dict()
import paddle
arg_1 = 1
arg_2 = 0.3
try:
  results["res_cpu"] = paddle.fft.fftfreq(arg_1,d=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.fft.fftfreq(arg_1,d=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
