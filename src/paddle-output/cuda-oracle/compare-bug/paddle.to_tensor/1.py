results = dict()
import paddle
arg_1_0 = -0.4
arg_1_1 = -0.2
arg_1_2 = 0.1
arg_1_3 = 0.3
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
try:
  results["res_cpu"] = paddle.to_tensor(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
try:
  results["res_gpu"] = paddle.to_tensor(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
