results = dict()
import paddle
arg_1_0 = 2
arg_1_1 = 8
arg_1_2 = 32
arg_1_3 = 32
arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
arg_2 = -39
arg_3 = -16
arg_class = paddle.nn.SpectralNorm(arg_1,dim=arg_2,power_iters=arg_3,)
arg_4_0_tensor = paddle.rand([2, 8, 32, 32], dtype=paddle.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
try:
  results["res_cpu"] = arg_class(*arg_4)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_4_0 = arg_4_0_tensor.clone().cuda()
arg_4 = [arg_4_0,]
try:
  results["res_gpu"] = arg_class(*arg_4)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
