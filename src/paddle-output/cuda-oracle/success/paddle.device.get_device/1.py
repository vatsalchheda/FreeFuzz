results = dict()
import paddle
try:
  results["res_cpu"] = paddle.device.get_device()
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.device.get_device()
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
