results = dict()
import paddle
arg_1_tensor = paddle.rand([33], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.0
try:
  results["res_cpu"] = paddle.clip(arg_1,min=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.clip(arg_1,min=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
