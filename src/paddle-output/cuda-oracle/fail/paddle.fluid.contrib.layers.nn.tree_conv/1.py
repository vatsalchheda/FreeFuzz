results = dict()
import paddle
arg_1_tensor = paddle.rand([-1, 10, 6], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([-1, 10, 2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = 3
arg_4 = 4
arg_5 = 2
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
