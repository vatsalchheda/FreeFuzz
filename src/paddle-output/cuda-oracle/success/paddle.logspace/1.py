results = dict()
import paddle
arg_1 = "replicate"
arg_2 = 10
arg_3 = -2
arg_4 = 2
arg_5 = "float32"
try:
  results["res_cpu"] = paddle.logspace(arg_1,arg_2,arg_3,arg_4,arg_5,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.logspace(arg_1,arg_2,arg_3,arg_4,arg_5,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
