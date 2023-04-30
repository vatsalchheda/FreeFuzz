results = dict()
import paddle
arg_1_tensor = paddle.randint(-4,32,[3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = True
try:
  results["res_cpu"] = paddle.linalg.cond(arg_1,p=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.linalg.cond(arg_1,p=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
