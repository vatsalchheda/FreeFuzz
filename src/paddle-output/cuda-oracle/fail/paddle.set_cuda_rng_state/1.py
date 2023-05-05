results = dict()
import paddle
arg_1 = []
try:
  results["res_cpu"] = paddle.set_cuda_rng_state(arg_1,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = []
try:
  results["res_gpu"] = paddle.set_cuda_rng_state(arg_1,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
