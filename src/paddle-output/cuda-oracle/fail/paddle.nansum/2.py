results = dict()
import paddle
arg_1_tensor = paddle.randint(-256,8192,[2, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3 = True
arg_4 = None
try:
  results["res_cpu"] = paddle.nansum(arg_1,axis=arg_2,keepdim=arg_3,name=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nansum(arg_1,axis=arg_2,keepdim=arg_3,name=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
