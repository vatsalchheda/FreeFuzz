results = dict()
import paddle
arg_1_0 = 1024
arg_1_1 = 66
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = "float32"
arg_3 = 0.0
arg_4 = 1.0
try:
  results["res_cpu"] = paddle.uniform(arg_1,dtype=arg_2,min=arg_3,max=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = [arg_1_0,arg_1_1,]
try:
  results["res_gpu"] = paddle.uniform(arg_1,dtype=arg_2,min=arg_3,max=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
