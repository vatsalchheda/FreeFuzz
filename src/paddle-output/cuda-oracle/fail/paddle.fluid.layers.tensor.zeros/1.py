results = dict()
import paddle
arg_1_0 = 2
arg_1_1 = 1
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = "float32"
try:
  results["res_cpu"] = paddle.fluid.layers.tensor.zeros(arg_1,dtype=arg_2,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = [arg_1_0,arg_1_1,]
try:
  results["res_gpu"] = paddle.fluid.layers.tensor.zeros(arg_1,dtype=arg_2,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
