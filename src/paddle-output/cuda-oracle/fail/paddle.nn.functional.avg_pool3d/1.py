results = dict()
import paddle
arg_1_tensor = paddle.rand([3, 1, 7, 112, 112], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 61.0
arg_2_1 = 62.0
arg_2_2 = False
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
arg_3 = -13
try:
  results["res_cpu"] = paddle.nn.functional.avg_pool3d(arg_1,kernel_size=arg_2,stride=arg_3,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
try:
  results["res_gpu"] = paddle.nn.functional.avg_pool3d(arg_1,kernel_size=arg_2,stride=arg_3,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
