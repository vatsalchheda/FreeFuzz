results = dict()
import paddle
arg_1 = -44
arg_2 = 0
arg_class = paddle.nn.MaxUnPool1D(kernel_size=arg_1,padding=arg_2,)
arg_3_0_tensor = paddle.rand([1, 3, 8], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.randint(-8192, 16384, [1, 3, 8], dtype=paddle.int32arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_cpu"] = arg_class(*arg_3)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_3_0 = arg_3_0_tensor.clone().cuda()
arg_3_1 = arg_3_1_tensor.clone().cuda()
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_gpu"] = arg_class(*arg_3)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
