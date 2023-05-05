results = dict()
import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = False
arg_4 = 1e-06
arg_5 = False
arg_6 = None
try:
  results["res_cpu"] = paddle.nn.functional.pairwise_distance(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.pairwise_distance(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
