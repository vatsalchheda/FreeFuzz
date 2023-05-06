results = dict()
import paddle
arg_1_tensor = paddle.rand([16, 164, 64], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1.0
arg_3 = True
try:
  results["res_cpu"] = paddle.fluid.layers.nn.scale(arg_1,arg_2,name=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.layers.nn.scale(arg_1,arg_2,name=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
