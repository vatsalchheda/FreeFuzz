results = dict()
import paddle
arg_1 = -64.0
arg_2_0 = 91
arg_2_1 = 1024
arg_2_2 = -27
arg_2_3 = 12
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
arg_3 = True
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
