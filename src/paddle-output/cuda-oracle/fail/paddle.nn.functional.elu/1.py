results = dict()
import paddle
arg_1_tensor = paddle.randint(-2048,32,[3, 3], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = 1.0
try:
  results["res_cpu"] = paddle.nn.functional.elu(arg_1,alpha=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.elu(arg_1,alpha=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
