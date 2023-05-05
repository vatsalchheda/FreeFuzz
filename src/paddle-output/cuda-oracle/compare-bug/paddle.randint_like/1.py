results = dict()
import paddle
arg_1_tensor = paddle.rand([128, 128, 11, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -5
arg_3 = 5
try:
  results["res_cpu"] = paddle.randint_like(arg_1,low=arg_2,high=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.randint_like(arg_1,low=arg_2,high=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
