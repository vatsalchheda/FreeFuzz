results = dict()
import paddle
arg_1_tensor = paddle.rand([6], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 7
arg_2_1 = 60
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = False
try:
  results["res_cpu"] = paddle.sum(arg_1,axis=arg_2,keepdim=arg_3,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_gpu"] = paddle.sum(arg_1,axis=arg_2,keepdim=arg_3,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
