results = dict()
import paddle
arg_1_tensor = paddle.rand([10, 10], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 0
arg_2_1 = 1
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_cpu"] = paddle.fft.ifftshift(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_gpu"] = paddle.fft.ifftshift(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
