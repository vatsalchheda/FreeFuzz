results = dict()
import paddle
arg_1_tensor = paddle.rand([2, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1024
arg_2 = [arg_2_0,]
arg_3 = False
arg_4 = None
try:
  results["res_cpu"] = paddle.nansum(arg_1,axis=arg_2,keepdim=arg_3,name=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,]
try:
  results["res_gpu"] = paddle.nansum(arg_1,axis=arg_2,keepdim=arg_3,name=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
