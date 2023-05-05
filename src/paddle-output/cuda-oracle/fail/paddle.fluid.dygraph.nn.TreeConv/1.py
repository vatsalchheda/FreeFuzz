results = dict()
import paddle
arg_1 = 5
arg_2 = 6
arg_3 = 1
arg_4 = 2
arg_class = paddle.fluid.dygraph.nn.TreeConv(feature_size=arg_1,output_size=arg_2,num_filters=arg_3,max_depth=arg_4,)
arg_5_0_tensor = paddle.rand([1, 10, 5], dtype=paddle.float32)
arg_5_0 = arg_5_0_tensor.clone()
arg_5_1_tensor = paddle.randint(-32768,1024,[1, 9, 2], dtype=paddle.int32)
arg_5_1 = arg_5_1_tensor.clone()
arg_5 = [arg_5_0,arg_5_1,]
try:
  results["res_cpu"] = arg_class(*arg_5)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_5_0 = arg_5_0_tensor.clone().cuda()
arg_5_1 = arg_5_1_tensor.clone().cuda()
arg_5 = [arg_5_0,arg_5_1,]
try:
  results["res_gpu"] = arg_class(*arg_5)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
