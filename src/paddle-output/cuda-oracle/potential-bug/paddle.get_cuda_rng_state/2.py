results = dict()
import paddle
try:
  results["res_cpu"] = paddle.get_cuda_rng_state()
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.get_cuda_rng_state()
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
