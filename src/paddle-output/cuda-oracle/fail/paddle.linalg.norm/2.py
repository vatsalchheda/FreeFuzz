results = dict()
import paddle
arg_1_tensor = paddle.randint(-128,64,[2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = inf
try:
  results["res_cpu"] = paddle.linalg.norm(arg_1,p=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.linalg.norm(arg_1,p=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
