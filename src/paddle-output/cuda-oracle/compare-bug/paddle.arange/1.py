results = dict()
import paddle
arg_1 = 0
arg_2 = 65
arg_3 = "float32"
arg_4 = None
try:
  results["res_cpu"] = paddle.arange(arg_1,arg_2,dtype=arg_3,name=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.arange(arg_1,arg_2,dtype=arg_3,name=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
