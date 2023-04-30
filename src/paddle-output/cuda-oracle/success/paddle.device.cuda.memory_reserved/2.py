results = dict()
import paddle
arg_1 = "gpu:0"
try:
  results["res_cpu"] = paddle.device.cuda.memory_reserved(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.device.cuda.memory_reserved(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
