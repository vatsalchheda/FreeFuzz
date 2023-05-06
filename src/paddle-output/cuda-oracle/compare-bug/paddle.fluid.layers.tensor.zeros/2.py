results = dict()
import paddle
arg_1_0 = 128
arg_1_1 = 128
arg_1_2 = 3
arg_1_3 = 1
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2 = "float32"
try:
  results["res_cpu"] = paddle.fluid.layers.tensor.zeros(arg_1,dtype=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
try:
  results["res_gpu"] = paddle.fluid.layers.tensor.zeros(arg_1,dtype=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
