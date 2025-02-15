results = dict()
import paddle
arg_1_tensor = paddle.randint(-16384, 128, [2, 2], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
try:
  results["res_cpu"] = paddle.fft.ifft2(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fft.ifft2(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
