results = dict()
import paddle
arg_1_tensor = paddle.randint(-8,4,[2, 2], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = -8.99
arg_3 = "max"
try:
  results["res_cpu"] = paddle.linalg.matrix_rank(arg_1,tol=arg_2,hermitian=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.linalg.matrix_rank(arg_1,tol=arg_2,hermitian=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
