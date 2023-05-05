results = dict()
import paddle
arg_1 = 3
arg_2 = -17
arg_3 = 66
arg_4 = "relu"
arg_class = paddle.fluid.dygraph.nn.Conv3D(num_channels=arg_1,num_filters=arg_2,filter_size=arg_3,act=arg_4,)
arg_5_0_tensor = paddle.rand([5, 3, 12, 32, 32], dtype=paddle.float32)
arg_5_0 = arg_5_0_tensor.clone()
arg_5 = [arg_5_0,]
try:
  results["res_cpu"] = arg_class(*arg_5)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_5_0 = arg_5_0_tensor.clone().cuda()
arg_5 = [arg_5_0,]
try:
  results["res_gpu"] = arg_class(*arg_5)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
