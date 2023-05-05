results = dict()
import paddle
arg_1_tensor = paddle.rand([2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
real = paddle.rand([3], paddle.float32)
imag = paddle.rand([3], paddle.float32)
arg_2_tensor = paddle.complex(real, imag)
arg_2 = arg_2_tensor.clone()
try:
  results["res_cpu"] = paddle.inner(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.inner(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
