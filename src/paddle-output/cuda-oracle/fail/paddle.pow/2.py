results = dict()
import paddle
arg_1_tensor = paddle.randint(-2048,8,[1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 18.0
try:
  results["res_cpu"] = paddle.pow(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.pow(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
