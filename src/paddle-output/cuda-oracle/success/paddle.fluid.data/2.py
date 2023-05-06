results = dict()
import paddle
arg_1 = False
arg_2_0 = -1
arg_2_1 = True
arg_2_2 = "zeros"
arg_2_3 = "replicate"
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
arg_3 = "float32"
try:
  results["res_cpu"] = paddle.fluid.data(name=arg_1,shape=arg_2,dtype=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
try:
  results["res_gpu"] = paddle.fluid.data(name=arg_1,shape=arg_2,dtype=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
