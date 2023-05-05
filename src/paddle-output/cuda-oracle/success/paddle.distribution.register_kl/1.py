results = dict()
import paddle
arg_1 = "Beta"
arg_2 = "Beta"
try:
  results["res_cpu"] = paddle.distribution.register_kl(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.distribution.register_kl(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
