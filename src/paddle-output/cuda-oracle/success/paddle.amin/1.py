results = dict()
import paddle
arg_1_tensor = paddle.rand([2, 2, 2], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = -60
arg_2_1 = 31
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_cpu"] = paddle.amin(arg_1,axis=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_gpu"] = paddle.amin(arg_1,axis=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
