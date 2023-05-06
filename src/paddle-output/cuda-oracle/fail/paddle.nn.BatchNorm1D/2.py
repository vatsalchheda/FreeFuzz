results = dict()
import paddle
arg_1 = -37.0
arg_2 = 1e-05
arg_3 = 0.9
arg_4 = -17
arg_5 = None
arg_6 = "NCL"
arg_7 = None
arg_class = paddle.nn.BatchNorm1D(arg_1,epsilon=arg_2,momentum=arg_3,weight_attr=arg_4,bias_attr=arg_5,data_format=arg_6,use_global_stats=arg_7,)
arg_8_0_tensor = paddle.rand([1, 1024, 961], dtype=paddle.float32)
arg_8_0 = arg_8_0_tensor.clone()
arg_8 = [arg_8_0,]
try:
  results["res_cpu"] = arg_class(*arg_8)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_8_0 = arg_8_0_tensor.clone().cuda()
arg_8 = [arg_8_0,]
try:
  results["res_gpu"] = arg_class(*arg_8)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
