results = dict()
import paddle
arg_1_tensor = paddle.rand([-1, 1536], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 4
arg_3 = -63.0
arg_4 = 1.0
try:
  results["res_cpu"] = paddle.histogram(arg_1,bins=arg_2,min=arg_3,max=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.histogram(arg_1,bins=arg_2,min=arg_3,max=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
