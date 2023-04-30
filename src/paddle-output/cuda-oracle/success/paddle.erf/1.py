results = dict()
import paddle
arg_1_tensor = paddle.randint(-4,16,[1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
try:
  results["res_cpu"] = paddle.erf(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.erf(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
