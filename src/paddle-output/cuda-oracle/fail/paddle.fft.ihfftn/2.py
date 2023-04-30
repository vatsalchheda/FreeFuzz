results = dict()
import paddle
arg_1_tensor = paddle.randint(-512,64,[1, 9, 2, 8, 5], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3_0 = 1
arg_3_1 = 2
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "backward"
arg_5 = -1089.0
try:
  results["res_cpu"] = paddle.fft.ihfftn(arg_1,arg_2,arg_3,arg_4,arg_5,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_gpu"] = paddle.fft.ihfftn(arg_1,arg_2,arg_3,arg_4,arg_5,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
