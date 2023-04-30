results = dict()
import paddle
arg_1 = 0.75
arg_2 = 51.391925438912004
arg_3 = 19
arg_4 = 44
try:
  results["res_cpu"] = paddle.linspace(arg_1,arg_2,arg_3,dtype=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.linspace(arg_1,arg_2,arg_3,dtype=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
