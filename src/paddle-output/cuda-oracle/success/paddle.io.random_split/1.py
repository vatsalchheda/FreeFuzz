results = dict()
import paddle
arg_1 = -36.0
arg_2_0 = -60.0
arg_2_1 = "zeros"
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_cpu"] = paddle.io.random_split(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_gpu"] = paddle.io.random_split(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
