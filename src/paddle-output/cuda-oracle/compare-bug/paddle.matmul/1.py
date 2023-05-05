results = dict()
import paddle
arg_1_tensor = paddle.rand([1, 2, 1, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([1, 2, 141, 4], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = True
try:
  results["res_cpu"] = paddle.matmul(x=arg_1,y=arg_2,transpose_y=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.matmul(x=arg_1,y=arg_2,transpose_y=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
