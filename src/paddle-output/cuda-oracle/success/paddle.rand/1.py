results = dict()
import paddle
arg_1_0 = -25
arg_1_1 = 1
arg_1_2 = 42
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
arg_2 = "float32"
try:
  results["res_cpu"] = paddle.rand(arg_1,dtype=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
try:
  results["res_gpu"] = paddle.rand(arg_1,dtype=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
