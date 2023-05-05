results = dict()
import paddle
arg_1 = 32
arg_2 = 2
arg_3 = "max"
arg_class = paddle.fluid.dygraph.nn.Conv2DTranspose(num_channels=arg_1,num_filters=arg_2,filter_size=arg_3,)
arg_4_0_tensor = paddle.rand([3, 32, 32, 5], dtype=paddle.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
try:
  results["res_cpu"] = arg_class(*arg_4)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_4_0 = arg_4_0_tensor.clone().cuda()
arg_4 = [arg_4_0,]
try:
  results["res_gpu"] = arg_class(*arg_4)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
