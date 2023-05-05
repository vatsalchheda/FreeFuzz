results = dict()
import paddle
arg_1_tensor = paddle.randint(-1,16384,[3, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4096,16384,[3, 5], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3 = -50.0
try:
  results["res_cpu"] = paddle.take(arg_1,arg_2,mode=arg_3,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.take(arg_1,arg_2,mode=arg_3,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
