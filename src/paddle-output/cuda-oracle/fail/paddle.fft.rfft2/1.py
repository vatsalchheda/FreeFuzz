results = dict()
import paddle
arg_1_tensor = paddle.randint(-16, 4, [1], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_0 = 4
arg_2_1 = 4
arg_2 = [arg_2_0,arg_2_1,]
arg_3_0 = 45
arg_3_1 = -55
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "backward"
try:
  results["res_cpu"] = paddle.fft.rfft2(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_gpu"] = paddle.fft.rfft2(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
