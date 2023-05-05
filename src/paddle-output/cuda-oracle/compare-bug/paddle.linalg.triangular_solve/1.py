results = dict()
import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 1], dtype=paddle.float64)
arg_2 = arg_2_tensor.clone()
arg_3 = False
try:
  results["res_cpu"] = paddle.linalg.triangular_solve(arg_1,arg_2,upper=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.linalg.triangular_solve(arg_1,arg_2,upper=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
