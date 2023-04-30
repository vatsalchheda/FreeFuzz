results = dict()
import paddle
arg_1_tensor = paddle.randint(-8192,512,[3, 2], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
try:
  results["res_cpu"] = paddle.linalg.qr(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.linalg.qr(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
