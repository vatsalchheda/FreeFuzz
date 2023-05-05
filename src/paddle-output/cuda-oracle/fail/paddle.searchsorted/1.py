results = dict()
import paddle
arg_1_tensor = paddle.randint(-1024,4,[2, 6], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = -52.0
arg_2_1 = -38.0
arg_2_2 = -12.0
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
try:
  results["res_cpu"] = paddle.searchsorted(arg_1,arg_2,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
try:
  results["res_gpu"] = paddle.searchsorted(arg_1,arg_2,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
