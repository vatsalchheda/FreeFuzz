results = dict()
import paddle
real = paddle.rand([5, 5], paddle.float64)
imag = paddle.rand([5, 5], paddle.float64)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 2
arg_2 = [arg_2_0,]
try:
  results["res_cpu"] = paddle.fft.fftshift(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,]
try:
  results["res_gpu"] = paddle.fft.fftshift(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
