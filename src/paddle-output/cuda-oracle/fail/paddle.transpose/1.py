results = dict()
import paddle
arg_1_tensor = paddle.randint(-4,256,[2, 3, 0], dtype=paddle.complex64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2_1 = 0
arg_2_2 = 2
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
try:
  results["res_cpu"] = paddle.transpose(arg_1,perm=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
try:
  results["res_gpu"] = paddle.transpose(arg_1,perm=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
