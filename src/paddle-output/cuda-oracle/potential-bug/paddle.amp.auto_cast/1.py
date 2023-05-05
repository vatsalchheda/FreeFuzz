results = dict()
import paddle
arg_1 = False
arg_2 = "O0"
try:
  results["res_cpu"] = paddle.amp.auto_cast(enable=arg_1,level=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.amp.auto_cast(enable=arg_1,level=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
