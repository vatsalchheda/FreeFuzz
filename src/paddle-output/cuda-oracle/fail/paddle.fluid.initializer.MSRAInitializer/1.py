results = dict()
import paddle
arg_1 = True
arg_2 = None
arg_3 = 55
arg_4 = 63.0
arg_5 = "leaky_relu"
arg_class = paddle.fluid.initializer.MSRAInitializer(uniform=arg_1,fan_in=arg_2,seed=arg_3,negative_slope=arg_4,nonlinearity=arg_5,)
arg_6_0_tensor = paddle.rand([512], dtype=paddle.float32)
arg_6_0 = arg_6_0_tensor.clone()
arg_6_1_tensor = paddle.rand([0, 2], dtype=paddle.float64)
arg_6_1 = arg_6_1_tensor.clone()
arg_6 = [arg_6_0,arg_6_1,]
try:
  results["res_cpu"] = arg_class(*arg_6)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_6_0 = arg_6_0_tensor.clone().cuda()
arg_6_1 = arg_6_1_tensor.clone().cuda()
arg_6 = [arg_6_0,arg_6_1,]
try:
  results["res_gpu"] = arg_class(*arg_6)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
