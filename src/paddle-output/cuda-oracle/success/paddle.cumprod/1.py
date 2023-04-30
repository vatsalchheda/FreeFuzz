results = dict()
import paddle
arg_1_tensor = paddle.randint(-4096,2048,[3, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = "circular"
arg_3 = "float64"
try:
  results["res_cpu"] = paddle.cumprod(arg_1,dim=arg_2,dtype=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.cumprod(arg_1,dim=arg_2,dtype=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
