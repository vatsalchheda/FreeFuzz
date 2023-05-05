results = dict()
import paddle
arg_1_tensor = paddle.randint(-2,32768,[3, 3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
try:
  results["res_cpu"] = paddle.unique(arg_1,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.unique(arg_1,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
