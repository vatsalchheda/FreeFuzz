results = dict()
import paddle
arg_1 = -34.0
arg_2_tensor = paddle.rand([2, 3, 2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([2, 2, 3], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
try:
  results["res_cpu"] = paddle.einsum(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.einsum(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
