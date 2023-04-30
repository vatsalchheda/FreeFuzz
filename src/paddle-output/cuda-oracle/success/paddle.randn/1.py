results = dict()
import paddle
arg_1_0 = -11
arg_1_1 = 54
arg_1_2 = 63
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
try:
  results["res_cpu"] = paddle.randn(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
try:
  results["res_gpu"] = paddle.randn(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
