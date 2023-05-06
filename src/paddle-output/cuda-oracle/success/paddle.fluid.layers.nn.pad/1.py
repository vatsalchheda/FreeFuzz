results = dict()
import paddle
arg_1_tensor = paddle.rand([-1, -1, 128], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = -22
arg_2_1 = -9
arg_2_2 = 9
arg_2_3 = -1
arg_2_4 = -1024
arg_2_5 = 59
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,arg_2_4,arg_2_5,]
try:
  results["res_cpu"] = paddle.fluid.layers.nn.pad(arg_1,paddings=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,arg_2_4,arg_2_5,]
try:
  results["res_gpu"] = paddle.fluid.layers.nn.pad(arg_1,paddings=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
