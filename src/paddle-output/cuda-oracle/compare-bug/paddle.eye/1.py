results = dict()
import paddle
arg_1 = 32
arg_2 = 57
arg_3 = "int32"
try:
  results["res_cpu"] = paddle.eye(arg_1,arg_2,dtype=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.eye(arg_1,arg_2,dtype=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
