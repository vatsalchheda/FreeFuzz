results = dict()
import paddle
arg_1_tensor = paddle.rand([3, 5], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
try:
  results["res_cpu"] = paddle.linalg.pinv(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.linalg.pinv(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
