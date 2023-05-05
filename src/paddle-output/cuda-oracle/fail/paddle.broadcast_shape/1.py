results = dict()
import paddle
arg_1_0 = 1
arg_1 = [arg_1_0,]
arg_2_0 = 1
arg_2 = [arg_2_0,]
try:
  results["res_cpu"] = paddle.broadcast_shape(arg_1,arg_2,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = [arg_1_0,]
arg_2 = [arg_2_0,]
try:
  results["res_gpu"] = paddle.broadcast_shape(arg_1,arg_2,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
