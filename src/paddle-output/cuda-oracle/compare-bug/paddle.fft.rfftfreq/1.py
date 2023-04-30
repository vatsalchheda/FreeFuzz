results = dict()
import paddle
arg_1 = 2
arg_2 = 53.3
try:
  results["res_cpu"] = paddle.fft.rfftfreq(arg_1,d=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.fft.rfftfreq(arg_1,d=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
