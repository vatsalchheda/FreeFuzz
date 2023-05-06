results = dict()
import paddle
arg_1 = False
try:
  results["res_cpu"] = paddle.jit.to_static(property=arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.jit.to_static(property=arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
