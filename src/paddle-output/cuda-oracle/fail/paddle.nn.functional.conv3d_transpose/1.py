results = dict()
import paddle
arg_1_tensor = paddle.rand([2, 4, 8, 8, 8], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([4, 6, 3, 3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([6], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = 41
arg_5 = 46
arg_6_0 = 1
arg_6_1 = 1
arg_6_2 = 1
arg_6 = [arg_6_0,arg_6_1,arg_6_2,]
arg_7_0 = -16
arg_7_1 = -18
arg_7_2 = -32
arg_7 = [arg_7_0,arg_7_1,arg_7_2,]
arg_8 = 1
arg_9 = None
arg_10 = "NCDHW"
try:
  results["res_cpu"] = paddle.nn.functional.conv3d_transpose(arg_1,arg_2,bias=arg_3,padding=arg_4,output_padding=arg_5,stride=arg_6,dilation=arg_7,groups=arg_8,output_size=arg_9,data_format=arg_10,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
arg_6 = [arg_6_0,arg_6_1,arg_6_2,]
arg_7 = [arg_7_0,arg_7_1,arg_7_2,]
try:
  results["res_gpu"] = paddle.nn.functional.conv3d_transpose(arg_1,arg_2,bias=arg_3,padding=arg_4,output_padding=arg_5,stride=arg_6,dilation=arg_7,groups=arg_8,output_size=arg_9,data_format=arg_10,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
