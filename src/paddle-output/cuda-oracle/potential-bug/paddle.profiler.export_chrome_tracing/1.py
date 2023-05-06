results = dict()
import paddle
arg_1 = "./log"
try:
  results["res_cpu"] = paddle.profiler.export_chrome_tracing(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.profiler.export_chrome_tracing(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
