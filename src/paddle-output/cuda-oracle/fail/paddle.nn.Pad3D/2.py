results = dict()
import paddle
arg_1_0 = -1
arg_1_1 = -16
arg_1_2 = -48
arg_1_3 = -37
arg_1_4 = 25
arg_1_5 = -24
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,arg_1_4,arg_1_5,]
arg_2 = "replicate"
arg_class = paddle.nn.Pad3D(padding=arg_1,mode=arg_2,)
arg_3_0_tensor = paddle.rand([1, 1, 1, 2, 3], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
try:
  results["res_cpu"] = arg_class(*arg_3)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_3_0 = arg_3_0_tensor.clone().cuda()
arg_3 = [arg_3_0,]
try:
  results["res_gpu"] = arg_class(*arg_3)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
