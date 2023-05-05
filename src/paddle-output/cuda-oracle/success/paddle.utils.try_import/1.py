results = dict()
import paddle
arg_1 = "regex"
try:
  results["res_cpu"] = paddle.utils.try_import(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.utils.try_import(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
