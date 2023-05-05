results = dict()
import paddle
arg_1 = -1080.0
try:
  results["res_cpu"] = paddle.set_default_dtype(arg_1,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.set_default_dtype(arg_1,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
