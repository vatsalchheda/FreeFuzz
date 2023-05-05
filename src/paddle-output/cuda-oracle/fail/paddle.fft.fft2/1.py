results = dict()
import paddle
float_tensor = paddle.rand([2, 2, 1], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_1_tensor = f16_tensor
arg_1 = arg_1_tensor.clone()
try:
  results["res_cpu"] = paddle.fft.fft2(arg_1,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fft.fft2(arg_1,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
