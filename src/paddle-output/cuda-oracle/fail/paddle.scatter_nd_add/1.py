results = dict()
import paddle
arg_1_tensor = paddle.rand([3, 5, 9, 10], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8,16384,[3, 2], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([3, 9, 10], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
try:
  results["res_cpu"] = paddle.scatter_nd_add(arg_1,arg_2,arg_3,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.scatter_nd_add(arg_1,arg_2,arg_3,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
