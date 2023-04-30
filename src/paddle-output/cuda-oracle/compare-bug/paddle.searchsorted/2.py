results = dict()
import paddle
arg_1_tensor = paddle.randint(-64,4096,[2, 6], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32,16384,[2, 4], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
arg_3 = True
try:
  results["res_cpu"] = paddle.searchsorted(arg_1,arg_2,right=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.searchsorted(arg_1,arg_2,right=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
