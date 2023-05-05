results = dict()
import paddle
try:
  results["res_cpu"] = paddle.utils.unique_name.guard()
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.utils.unique_name.guard()
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
