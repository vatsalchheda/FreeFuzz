results = dict()
import paddle
arg_1_tensor = paddle.randint(-8192, 512, [3], dtype=paddle.int32arg_1 = arg_1_tensor.clone()
arg_2_0 = 2
arg_2 = [arg_2_0,]
try:
  results["res_cpu"] = paddle.expand(arg_1,shape=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,]
try:
  results["res_gpu"] = paddle.expand(arg_1,shape=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
