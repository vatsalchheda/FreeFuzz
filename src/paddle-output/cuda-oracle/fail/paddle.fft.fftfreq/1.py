results = dict()
import paddle
arg_1 = 20
arg_2 = -1086.0
arg_3 = "float64"
try:
  results["res_cpu"] = paddle.fft.fftfreq(arg_1,arg_2,arg_3,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.fft.fftfreq(arg_1,arg_2,arg_3,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
