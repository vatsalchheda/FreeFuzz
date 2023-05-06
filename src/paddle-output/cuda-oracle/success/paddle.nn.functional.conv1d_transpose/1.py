results = dict()
import paddle
arg_1_tensor = paddle.rand([1, 512, 136], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([512, 256, 10], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([256], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = None
arg_5 = 1
arg_6 = 1024
arg_7_0 = 5
arg_7 = [arg_7_0,]
arg_8_0 = 1
arg_8 = [arg_8_0,]
arg_9 = 35
arg_10 = "NCL"
try:
  results["res_cpu"] = paddle.nn.functional.conv1d_transpose(arg_1,arg_2,bias=arg_3,output_size=arg_4,output_padding=arg_5,padding=arg_6,stride=arg_7,dilation=arg_8,groups=arg_9,data_format=arg_10,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
arg_7 = [arg_7_0,]
arg_8 = [arg_8_0,]
try:
  results["res_gpu"] = paddle.nn.functional.conv1d_transpose(arg_1,arg_2,bias=arg_3,output_size=arg_4,output_padding=arg_5,padding=arg_6,stride=arg_7,dilation=arg_8,groups=arg_9,data_format=arg_10,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
