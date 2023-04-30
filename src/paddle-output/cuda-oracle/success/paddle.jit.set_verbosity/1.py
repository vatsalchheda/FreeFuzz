results = dict()
import paddle
arg_1 = 52
try:
  results["res_cpu"] = paddle.jit.set_verbosity(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.jit.set_verbosity(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
