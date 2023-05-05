results = dict()
import paddle
real = paddle.rand([-1, 10, 1024], paddle.float64)
imag = paddle.rand([-1, 10, 1024], paddle.float64)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([-1, 10, 2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = 0
arg_4 = -1024
arg_5 = -30
try:
  results["res_cpu"] = paddle.fluid.contrib.layers.nn.tree_conv(arg_1,arg_2,arg_3,arg_4,arg_5,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.contrib.layers.nn.tree_conv(arg_1,arg_2,arg_3,arg_4,arg_5,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
