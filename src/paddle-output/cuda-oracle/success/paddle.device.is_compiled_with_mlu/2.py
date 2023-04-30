results = dict()
import paddle
try:
  results["res_cpu"] = paddle.device.is_compiled_with_mlu()
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.device.is_compiled_with_mlu()
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
