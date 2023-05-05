results = dict()
import paddle
arg_1_0 = 10
arg_1_1 = 10
arg_1 = [arg_1_0,arg_1_1,]
try:
  results["res_cpu"] = paddle.uniform(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = [arg_1_0,arg_1_1,]
try:
  results["res_gpu"] = paddle.uniform(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
