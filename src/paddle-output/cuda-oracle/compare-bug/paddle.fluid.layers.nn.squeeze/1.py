results = dict()
import paddle
real = paddle.rand([-1, 0, 128], paddle.float64)
imag = paddle.rand([-1, 0, 128], paddle.float64)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2 = [arg_2_0,]
try:
  results["res_cpu"] = paddle.fluid.layers.nn.squeeze(arg_1,axes=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,]
try:
  results["res_gpu"] = paddle.fluid.layers.nn.squeeze(arg_1,axes=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
