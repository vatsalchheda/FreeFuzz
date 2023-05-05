results = dict()
import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4,256,[2], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
arg_3 = 1
arg_4_tensor = paddle.rand([2, 3], dtype=paddle.float32)
arg_4 = arg_4_tensor.clone()
try:
  results["res_cpu"] = paddle.index_add(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_4 = arg_4_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.index_add(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
