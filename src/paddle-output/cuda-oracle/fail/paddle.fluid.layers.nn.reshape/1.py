results = dict()
import paddle
arg_1_tensor = paddle.rand([32, 32, 3, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 32
arg_2_1 = 32
arg_2_2 = 3
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
arg_3 = "Normal_sample"
try:
  results["res_cpu"] = paddle.fluid.layers.nn.reshape(arg_1,arg_2,name=arg_3,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
try:
  results["res_gpu"] = paddle.fluid.layers.nn.reshape(arg_1,arg_2,name=arg_3,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
