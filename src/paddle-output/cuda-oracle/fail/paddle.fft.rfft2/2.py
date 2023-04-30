results = dict()
import paddle
arg_1_tensor = paddle.randint(-64,4,[9, 2, 6, 6], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3_0 = 0
arg_3_1 = -3
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "ortho"
try:
  results["res_cpu"] = paddle.fft.rfft2(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_gpu"] = paddle.fft.rfft2(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
