results = dict()
import paddle
arg_1_0 = 59
arg_1_1 = 13
arg_1_2 = 62
arg_1_3 = -27
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2 = -67
try:
  results["res_cpu"] = paddle.zeros(shape=arg_1,dtype=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
try:
  results["res_gpu"] = paddle.zeros(shape=arg_1,dtype=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
