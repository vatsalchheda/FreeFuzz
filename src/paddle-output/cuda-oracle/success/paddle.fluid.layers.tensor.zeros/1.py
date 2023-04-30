results = dict()
import paddle
arg_1_0 = 3
arg_1 = [arg_1_0,]
arg_2 = 38.0
try:
  results["res_cpu"] = paddle.fluid.layers.tensor.zeros(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = [arg_1_0,]
try:
  results["res_gpu"] = paddle.fluid.layers.tensor.zeros(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
