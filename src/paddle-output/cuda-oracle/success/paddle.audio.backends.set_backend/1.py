results = dict()
import paddle
arg_1 = 0
try:
  results["res_cpu"] = paddle.audio.backends.set_backend(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.audio.backends.set_backend(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
