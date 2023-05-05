results = dict()
import paddle
arg_1_0_tensor = paddle.randint(-8, 1024, [1], dtype=paddle.int32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
try:
  results["res_cpu"] = paddle.stack(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1_0 = arg_1_0_tensor.clone().cuda()
arg_1 = [arg_1_0,]
try:
  results["res_gpu"] = paddle.stack(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
