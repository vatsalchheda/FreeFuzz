results = dict()
import paddle
arg_1_0 = 256
arg_1_1 = 256
arg_1_2 = 11
arg_1_3 = 1
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2 = 0.0
arg_3 = 1.0
arg_4 = 0
arg_5 = "float32"
try:
  results["res_cpu"] = paddle.fluid.layers.nn.gaussian_random(arg_1,mean=arg_2,std=arg_3,seed=arg_4,dtype=arg_5,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
try:
  results["res_gpu"] = paddle.fluid.layers.nn.gaussian_random(arg_1,mean=arg_2,std=arg_3,seed=arg_4,dtype=arg_5,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
