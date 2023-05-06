results = dict()
import paddle
real = paddle.rand([1, 30001], paddle.float32)
imag = paddle.rand([1, 30001], paddle.float32)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = 5
try:
  results["res_cpu"] = paddle.topk(arg_1,k=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.topk(arg_1,k=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
