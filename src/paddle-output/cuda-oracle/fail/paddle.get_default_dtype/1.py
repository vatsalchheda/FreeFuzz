results = dict()
import paddle
try:
  results["res_cpu"] = paddle.get_default_dtype()
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.get_default_dtype()
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
