results = dict()
import paddle
arg_1_tensor = paddle.rand([], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0.67
arg_3 = 29.72
try:
  results["res_cpu"] = paddle.stanh(arg_1,scale_a=arg_2,scale_b=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.stanh(arg_1,scale_a=arg_2,scale_b=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
