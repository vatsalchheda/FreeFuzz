results = dict()
import paddle
arg_1 = 36
arg_2 = 0
try:
  results["res_cpu"] = paddle.static.ipu_shard_guard(index=arg_1,stage=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.static.ipu_shard_guard(index=arg_1,stage=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
