results = dict()
import paddle
arg_1 = "cpu"
try:
  results["res_cpu"] = paddle.disable_static(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.disable_static(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
