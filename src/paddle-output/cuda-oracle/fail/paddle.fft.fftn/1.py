results = dict()
import paddle
arg_1_tensor = paddle.rand([1, 4, 2, 6, 9], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = None
arg_4 = "backward"
arg_5 = None
try:
  results["res_cpu"] = paddle.fft.fftn(arg_1,arg_2,arg_3,arg_4,arg_5,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fft.fftn(arg_1,arg_2,arg_3,arg_4,arg_5,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
