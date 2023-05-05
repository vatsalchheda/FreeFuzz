results = dict()
import paddle
arg_1 = "train"
try:
  results["res_cpu"] = paddle.distributed.spawn(arg_1,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.distributed.spawn(arg_1,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
