results = dict()
import paddle
arg_1_tensor = paddle.randint(0,2,[2, 2], dtype=paddle.bool)
arg_1 = arg_1_tensor.clone()
arg_2 = -46
try:
  results["res_cpu"] = paddle.all(arg_1,axis=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.all(arg_1,axis=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
