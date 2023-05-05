results = dict()
import paddle
real = paddle.rand([4, 4, 4], paddle.float64)
imag = paddle.rand([4, 4, 4], paddle.float64)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = True
arg_4 = "backward"
try:
  results["res_cpu"] = paddle.fft.irfft(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fft.irfft(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
