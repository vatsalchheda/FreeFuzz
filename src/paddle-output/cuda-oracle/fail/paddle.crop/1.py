results = dict()
import paddle
arg_1_tensor = paddle.randint(-512,32768,[3, 3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4096,2048,[2], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-32,8192,[2], dtype=paddle.int32)
arg_3 = arg_3_tensor.clone()
try:
  results["res_cpu"] = paddle.crop(arg_1,arg_2,arg_3,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.crop(arg_1,arg_2,arg_3,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
