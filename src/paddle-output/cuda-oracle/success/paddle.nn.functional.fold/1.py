results = dict()
import paddle
arg_1_tensor = paddle.rand([2, 12, 12], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 4
arg_2_1 = 5
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = 1e+20
try:
  results["res_cpu"] = paddle.nn.functional.fold(arg_1,output_sizes=arg_2,kernel_sizes=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_gpu"] = paddle.nn.functional.fold(arg_1,output_sizes=arg_2,kernel_sizes=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
