results = dict()
import paddle
arg_1_tensor = paddle.randint(-8,64,[13], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 4
arg_3 = -1
try:
  results["res_cpu"] = paddle.triu_indices(arg_1,arg_2,arg_3,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.triu_indices(arg_1,arg_2,arg_3,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
