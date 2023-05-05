results = dict()
import paddle
try:
  results["res_cpu"] = paddle.disable_signal_handler()
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.disable_signal_handler()
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
