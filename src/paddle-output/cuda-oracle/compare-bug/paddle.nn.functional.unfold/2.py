results = dict()
import paddle
arg_1_tensor = paddle.rand([100, 3, 224, 224], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 3
arg_2_1 = 3
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = 1
arg_4 = 1
arg_5 = 1
try:
  results["res_cpu"] = paddle.nn.functional.unfold(arg_1,arg_2,arg_3,arg_4,arg_5,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_gpu"] = paddle.nn.functional.unfold(arg_1,arg_2,arg_3,arg_4,arg_5,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
