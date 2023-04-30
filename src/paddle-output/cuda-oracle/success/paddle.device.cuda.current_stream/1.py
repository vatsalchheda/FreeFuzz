results = dict()
import paddle
try:
  results["res_cpu"] = paddle.device.cuda.current_stream()
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.device.cuda.current_stream()
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
