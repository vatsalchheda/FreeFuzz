results = dict()
import paddle
arg_1 = -23
arg_2 = 4
arg_3 = -72
try:
  results["res_cpu"] = paddle.triu_indices(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.triu_indices(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
