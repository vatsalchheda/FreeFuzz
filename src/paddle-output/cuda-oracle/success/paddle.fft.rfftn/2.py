results = dict()
import paddle
arg_1_tensor = paddle.rand([3, 4, 4, 3, 4], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 16
arg_2_1 = -40
arg_2 = [arg_2_0,arg_2_1,]
arg_3_0 = 1
arg_3_1 = 2
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "backward"
try:
  results["res_cpu"] = paddle.fft.rfftn(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_gpu"] = paddle.fft.rfftn(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
