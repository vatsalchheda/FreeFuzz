results = dict()
import paddle
arg_1_tensor = paddle.rand([2, 3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2_1 = 2
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_cpu"] = paddle.count_nonzero(arg_1,axis=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_gpu"] = paddle.count_nonzero(arg_1,axis=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
