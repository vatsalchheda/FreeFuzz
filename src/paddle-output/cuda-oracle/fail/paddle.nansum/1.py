results = dict()
import paddle
arg_1_tensor = paddle.rand([2, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
try:
  results["res_cpu"] = paddle.nansum(arg_1,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nansum(arg_1,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
