results = dict()
import paddle
arg_1_tensor = paddle.randint(-2,256,[13], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 8
arg_3 = 2
try:
  results["res_cpu"] = paddle.tril_indices(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.tril_indices(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
