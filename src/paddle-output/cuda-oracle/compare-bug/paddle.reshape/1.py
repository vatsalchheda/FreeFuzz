results = dict()
import paddle
arg_1_tensor = paddle.randint(-4096,128,[6], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 2
arg_2_1 = 3
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = False
try:
  results["res_cpu"] = paddle.reshape(arg_1,arg_2,name=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_gpu"] = paddle.reshape(arg_1,arg_2,name=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
