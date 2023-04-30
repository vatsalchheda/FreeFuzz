results = dict()
import paddle
arg_1_tensor = paddle.randint(-16384,32768,[2, 16], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.8
arg_3 = 0
try:
  results["res_cpu"] = paddle.quantile(arg_1,q=arg_2,axis=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.quantile(arg_1,q=arg_2,axis=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
