results = dict()
import paddle
arg_1_0 = 48
arg_1 = [arg_1_0,]
try:
  results["res_cpu"] = paddle.rand(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = [arg_1_0,]
try:
  results["res_gpu"] = paddle.rand(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
