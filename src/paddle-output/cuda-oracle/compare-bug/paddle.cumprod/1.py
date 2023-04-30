results = dict()
import paddle
arg_1_tensor = paddle.randint(-1024,1024,[13, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = -1
try:
  results["res_cpu"] = paddle.cumprod(arg_1,dim=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.cumprod(arg_1,dim=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
