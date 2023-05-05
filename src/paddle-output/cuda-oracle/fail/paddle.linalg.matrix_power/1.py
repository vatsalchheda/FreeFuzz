results = dict()
import paddle
arg_1_tensor = paddle.rand([3, 1024], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = -55
try:
  results["res_cpu"] = paddle.linalg.matrix_power(arg_1,arg_2,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.linalg.matrix_power(arg_1,arg_2,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
