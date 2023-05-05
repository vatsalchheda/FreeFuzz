results = dict()
import paddle
real = paddle.rand([8, 6, 5, 8, 7], paddle.float64)
imag = paddle.rand([8, 6, 5, 8, 7], paddle.float64)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = -1
arg_4 = "reflect"
try:
  results["res_cpu"] = paddle.fft.ifft(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fft.ifft(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
