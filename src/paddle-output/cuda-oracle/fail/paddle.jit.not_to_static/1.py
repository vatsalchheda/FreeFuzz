results = dict()
import paddle
arg_1 = "recompute_training"
try:
  results["res_cpu"] = paddle.jit.not_to_static(arg_1,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.jit.not_to_static(arg_1,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
