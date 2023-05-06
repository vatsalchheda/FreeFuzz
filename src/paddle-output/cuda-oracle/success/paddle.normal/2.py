results = dict()
import paddle
arg_1 = 44.0
arg_2 = -46.98
arg_3_0 = "max"
arg_3_1 = -64.0
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_cpu"] = paddle.normal(mean=arg_1,std=arg_2,shape=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_gpu"] = paddle.normal(mean=arg_1,std=arg_2,shape=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
