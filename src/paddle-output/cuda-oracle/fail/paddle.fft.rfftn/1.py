results = dict()
import paddle
arg_1_tensor = paddle.rand([5, 1, 2, 3, 8], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3_0 = 0
arg_3_1 = 1
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "backward"
arg_5 = None
try:
  results["res_cpu"] = paddle.fft.rfftn(arg_1,arg_2,arg_3,arg_4,arg_5,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_gpu"] = paddle.fft.rfftn(arg_1,arg_2,arg_3,arg_4,arg_5,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
