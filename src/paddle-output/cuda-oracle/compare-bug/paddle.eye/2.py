results = dict()
import paddle
arg_1 = 17
arg_2 = "float32"
try:
  results["res_cpu"] = paddle.eye(arg_1,dtype=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
try:
  results["res_gpu"] = paddle.eye(arg_1,dtype=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
