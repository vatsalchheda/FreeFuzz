results = dict()
import paddle
arg_1 = -1.0
arg_2 = -2.9800000000000004
arg_3_0 = -27
arg_3_1 = 66
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
