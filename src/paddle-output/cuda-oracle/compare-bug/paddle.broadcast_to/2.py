results = dict()
import paddle
arg_1_tensor = paddle.randint(-8, 8, [1, 5], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2_1 = 5
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_cpu"] = paddle.broadcast_to(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_gpu"] = paddle.broadcast_to(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
