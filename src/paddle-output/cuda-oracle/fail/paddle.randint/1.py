results = dict()
import paddle
arg_1 = -8
arg_2 = 1024
arg_3_0 = -16
arg_3 = [arg_3_0,]
arg_4 = "paddleVarType"
try:
  results["res_cpu"] = paddle.randint(arg_1,arg_2,arg_3,dtype=arg_4,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_3 = [arg_3_0,]
try:
  results["res_gpu"] = paddle.randint(arg_1,arg_2,arg_3,dtype=arg_4,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
