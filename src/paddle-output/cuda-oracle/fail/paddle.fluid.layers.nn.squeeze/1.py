results = dict()
import paddle
arg_1_tensor = paddle.randint(-1,64,[10, 4, 1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 2
arg_2 = [arg_2_0,]
try:
  results["res_cpu"] = paddle.fluid.layers.nn.squeeze(arg_1,arg_2,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,]
try:
  results["res_gpu"] = paddle.fluid.layers.nn.squeeze(arg_1,arg_2,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
