results = dict()
import paddle
arg_1_tensor = paddle.randint(-128, 4, [4, 1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2_1 = 4
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_cpu"] = paddle.tile(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_gpu"] = paddle.tile(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
