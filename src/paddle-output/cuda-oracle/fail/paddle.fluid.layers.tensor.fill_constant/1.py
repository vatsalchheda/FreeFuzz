results = dict()
import paddle
arg_1 = 0.0
arg_2_0 = 256
arg_2_1 = 256
arg_2_2 = 11
arg_2_3 = 1
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
arg_3 = "float32"
arg_4 = False
arg_5 = None
try:
  results["res_cpu"] = paddle.fluid.layers.tensor.fill_constant(value=arg_1,shape=arg_2,dtype=arg_3,force_cpu=arg_4,name=arg_5,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
try:
  results["res_gpu"] = paddle.fluid.layers.tensor.fill_constant(value=arg_1,shape=arg_2,dtype=arg_3,force_cpu=arg_4,name=arg_5,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
