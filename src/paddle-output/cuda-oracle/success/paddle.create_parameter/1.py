results = dict()
import paddle
arg_1_0 = 29
arg_1 = [arg_1_0,]
arg_2 = "float32"
try:
  results["res_cpu"] = paddle.create_parameter(shape=arg_1,dtype=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = [arg_1_0,]
try:
  results["res_gpu"] = paddle.create_parameter(shape=arg_1,dtype=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
