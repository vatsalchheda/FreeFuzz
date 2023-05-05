results = dict()
import paddle
real = paddle.rand([8, 257, 376], paddle.float64)
imag = paddle.rand([8, 257, 376], paddle.float64)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = -16
try:
  results["res_cpu"] = paddle.signal.istft(arg_1,n_fft=arg_2,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.signal.istft(arg_1,n_fft=arg_2,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
