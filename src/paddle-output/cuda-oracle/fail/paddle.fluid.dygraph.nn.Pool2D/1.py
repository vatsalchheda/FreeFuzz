results = dict()
import paddle
arg_1 = -36
arg_2 = "max"
arg_3 = 58
arg_4 = False
arg_class = paddle.fluid.dygraph.nn.Pool2D(pool_size=arg_1,pool_type=arg_2,pool_stride=arg_3,global_pooling=arg_4,)
arg_5_0_tensor = paddle.rand([3, 32, 32, 5], dtype=paddle.float32)
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
