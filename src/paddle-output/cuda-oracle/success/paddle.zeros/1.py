results = dict()
import paddle
arg_1_0 = 3
arg_1_1 = -1024
arg_1_2 = 36
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
arg_2 = "paddleVarType"
try:
  results["res_cpu"] = paddle.zeros(arg_1,dtype=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
try:
  results["res_gpu"] = paddle.zeros(arg_1,dtype=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
