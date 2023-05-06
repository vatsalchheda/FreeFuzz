results = dict()
import paddle
arg_1_tensor = paddle.rand([1, 128, 967], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([128, 128, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([128], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = 0
arg_5 = 2
arg_6_0 = 3
arg_6 = [arg_6_0,]
arg_7 = 1
arg_8 = "mean"
try:
  results["res_cpu"] = paddle.nn.functional.conv1d(arg_1,arg_2,bias=arg_3,padding=arg_4,stride=arg_5,dilation=arg_6,groups=arg_7,data_format=arg_8,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
arg_6 = [arg_6_0,]
try:
  results["res_gpu"] = paddle.nn.functional.conv1d(arg_1,arg_2,bias=arg_3,padding=arg_4,stride=arg_5,dilation=arg_6,groups=arg_7,data_format=arg_8,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
