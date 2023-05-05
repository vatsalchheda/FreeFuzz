results = dict()
import paddle
arg_1_tensor = paddle.rand([0, 32, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
try:
  results["res_cpu"] = paddle.is_empty(x=arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.is_empty(x=arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
