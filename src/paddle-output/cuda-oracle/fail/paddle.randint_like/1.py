results = dict()
import paddle
arg_1_tensor = paddle.randint(-4096,16384,[1, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
arg_3 = 5
arg_4 = "int32"
try:
  results["res_cpu"] = paddle.randint_like(arg_1,low=arg_2,high=arg_3,dtype=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.randint_like(arg_1,low=arg_2,high=arg_3,dtype=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
