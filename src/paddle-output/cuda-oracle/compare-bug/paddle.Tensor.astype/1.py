results = dict()
import paddle
arg_1_tensor = paddle.randint(-32,16384,[2, 1], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2 = "float64"
try:
  results["res_cpu"] = paddle.Tensor.astype(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.Tensor.astype(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
