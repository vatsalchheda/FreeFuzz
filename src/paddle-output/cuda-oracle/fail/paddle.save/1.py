results = dict()
import paddle
arg_1 = "collectionsOrderedDict"
arg_2 = "C:\Users\phalt\AppData\Local\Temp\tmpn5rk4ak3\model.pdparams"
try:
  results["res_cpu"] = paddle.save(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.save(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
