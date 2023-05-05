results = dict()
import paddle
real = paddle.rand([4, 4, 4], paddle.float64)
imag = paddle.rand([4, 4, 4], paddle.float64)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3_0 = -1024.0
arg_3_1 = True
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "forward"
try:
  results["res_cpu"] = paddle.fft.hfft2(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_gpu"] = paddle.fft.hfft2(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
