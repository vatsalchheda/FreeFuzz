results = dict()
import paddle
arg_1 = 0
arg_2 = 12207.0
arg_3 = 66
arg_4 = "float32"
try:
  results["res_cpu"] = paddle.linspace(arg_1,arg_2,arg_3,dtype=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.linspace(arg_1,arg_2,arg_3,dtype=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
