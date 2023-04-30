results = dict()
import paddle
arg_1_tensor = paddle.randint(0,2,[2, 1], dtype=paddle.bool)
arg_1 = arg_1_tensor.clone()
try:
  results["res_cpu"] = paddle.poisson(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.poisson(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
