results = dict()
import paddle
arg_1_tensor = paddle.randint(-512,8192,[1], dtype=paddle.bfloat16)
arg_1 = arg_1_tensor.clone()
try:
  results["res_cpu"] = paddle.Tensor.item(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.Tensor.item(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
