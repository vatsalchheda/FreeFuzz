results = dict()
import paddle
arg_1 = -35
arg_2_0 = 8
arg_2_1 = 20
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = "int32"
try:
  results["res_cpu"] = paddle.fluid.data(name=arg_1,shape=arg_2,dtype=arg_3,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_gpu"] = paddle.fluid.data(name=arg_1,shape=arg_2,dtype=arg_3,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
