results = dict()
import paddle
arg_1_0 = 2
arg_1_1 = 1
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = 0
try:
  results["res_cpu"] = paddle.fluid.layers.nn.uniform_random(arg_1,seed=arg_2,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = [arg_1_0,arg_1_1,]
try:
  results["res_gpu"] = paddle.fluid.layers.nn.uniform_random(arg_1,seed=arg_2,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
