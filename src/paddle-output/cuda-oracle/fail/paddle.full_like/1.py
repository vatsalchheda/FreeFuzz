results = dict()
import paddle
arg_1_tensor = paddle.rand([10, 8, 22, 123], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -1024.0
try:
  results["res_cpu"] = paddle.full_like(arg_1,fill_value=arg_2,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.full_like(arg_1,fill_value=arg_2,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
