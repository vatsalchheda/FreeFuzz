results = dict()
import paddle
real = paddle.rand([0, 3, 3], paddle.float32)
imag = paddle.rand([0, 3, 3], paddle.float32)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
try:
  results["res_cpu"] = paddle.linalg.slogdet(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.linalg.slogdet(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
