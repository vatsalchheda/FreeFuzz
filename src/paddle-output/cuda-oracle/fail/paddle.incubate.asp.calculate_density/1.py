results = dict()
import paddle
arg_1_tensor = paddle.randint(-512, 8192, [2, 4], dtype=paddle.int32arg_1 = arg_1_tensor.clone()
try:
  results["res_cpu"] = paddle.incubate.asp.calculate_density(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.incubate.asp.calculate_density(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
