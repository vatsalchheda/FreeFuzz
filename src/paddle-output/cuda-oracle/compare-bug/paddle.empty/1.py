results = dict()
import paddle
arg_1_0 = 46
arg_1_1 = 62
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = "float32"
try:
  results["res_cpu"] = paddle.empty(shape=arg_1,dtype=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = [arg_1_0,arg_1_1,]
try:
  results["res_gpu"] = paddle.empty(shape=arg_1,dtype=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
