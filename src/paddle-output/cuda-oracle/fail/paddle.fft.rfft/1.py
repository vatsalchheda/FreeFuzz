results = dict()
import paddle
arg_1_tensor = paddle.rand([8, 5, 6, 9, 6], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = 3
arg_3 = -1
arg_4 = "backward"
try:
  results["res_cpu"] = paddle.fft.rfft(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fft.rfft(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
