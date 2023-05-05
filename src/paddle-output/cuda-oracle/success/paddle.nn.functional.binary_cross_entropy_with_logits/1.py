results = dict()
import paddle
real = paddle.rand([3], paddle.float64)
imag = paddle.rand([3], paddle.float64)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = None
arg_4 = "mean"
arg_5 = None
arg_6 = None
try:
  results["res_cpu"] = paddle.nn.functional.binary_cross_entropy_with_logits(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.binary_cross_entropy_with_logits(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
