results = dict()
import paddle
arg_1 = 20
arg_2 = 0.5
arg_3 = "float32"
try:
  results["res_cpu"] = paddle.fft.rfftfreq(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.fft.rfftfreq(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
