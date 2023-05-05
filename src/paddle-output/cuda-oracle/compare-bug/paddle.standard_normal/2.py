results = dict()
import paddle
arg_1_0 = 256
arg_1_1 = 256
arg_1_2 = 3
arg_1_3 = 1
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
try:
  results["res_cpu"] = paddle.standard_normal(shape=arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
try:
  results["res_gpu"] = paddle.standard_normal(shape=arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
