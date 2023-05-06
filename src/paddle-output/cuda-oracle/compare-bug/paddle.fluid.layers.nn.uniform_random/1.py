results = dict()
import paddle
arg_1_0 = 2
arg_1_1 = 1
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = "float32"
arg_3 = -51.0
arg_4 = -78
arg_5 = 53
try:
  results["res_cpu"] = paddle.fluid.layers.nn.uniform_random(arg_1,dtype=arg_2,min=arg_3,max=arg_4,seed=arg_5,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = [arg_1_0,arg_1_1,]
try:
  results["res_gpu"] = paddle.fluid.layers.nn.uniform_random(arg_1,dtype=arg_2,min=arg_3,max=arg_4,seed=arg_5,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
