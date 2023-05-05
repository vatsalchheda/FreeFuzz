results = dict()
import paddle
arg_1 = 4
arg_2 = 114
arg_3 = 23.0
try:
  results["res_cpu"] = paddle.set_printoptions(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.set_printoptions(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
