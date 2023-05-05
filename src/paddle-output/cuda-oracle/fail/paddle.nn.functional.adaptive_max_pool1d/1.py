results = dict()
import paddle
arg_1_tensor = paddle.rand([1, 3, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
arg_3 = 56
arg_4 = None
try:
  results["res_cpu"] = paddle.nn.functional.adaptive_max_pool1d(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.adaptive_max_pool1d(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
