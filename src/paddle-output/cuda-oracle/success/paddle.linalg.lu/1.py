results = dict()
import paddle
real = paddle.rand([1024, 0], paddle.float64)
imag = paddle.rand([1024, 0], paddle.float64)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = True
try:
  results["res_cpu"] = paddle.linalg.lu(arg_1,get_infos=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.linalg.lu(arg_1,get_infos=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
