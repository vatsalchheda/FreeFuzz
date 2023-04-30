results = dict()
import paddle
arg_1 = "C:\Users\phalt\.cache\ppgan\GPEN-512.pdparams"
try:
  results["res_cpu"] = paddle.load(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.load(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
