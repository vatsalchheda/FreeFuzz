results = dict()
import paddle
arg_1_tensor = paddle.randint(-32768,8,[2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -44
arg_3 = 1
try:
  results["res_cpu"] = paddle.moveaxis(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.moveaxis(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
