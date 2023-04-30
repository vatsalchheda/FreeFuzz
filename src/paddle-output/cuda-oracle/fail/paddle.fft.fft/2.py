results = dict()
import paddle
arg_1_tensor = paddle.randint(-16,16384,[7], dtype=paddle.complex128)
arg_1 = arg_1_tensor.clone()
try:
  results["res_cpu"] = paddle.fft.fft(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fft.fft(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
