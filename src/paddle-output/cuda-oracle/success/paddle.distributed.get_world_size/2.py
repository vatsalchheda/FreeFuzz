results = dict()
import paddle
try:
  results["res_cpu"] = paddle.distributed.get_world_size()
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.distributed.get_world_size()
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
