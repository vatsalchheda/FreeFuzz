results = dict()
import paddle
try:
  results["res_cpu"] = paddle.audio.backends.get_current_backend()
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.audio.backends.get_current_backend()
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
