results = dict()
import paddle
arg_1_tensor = paddle.randint(-2048, 8192, [2, 2], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2 = None
arg_3_0 = -32
arg_3_1 = -57
arg_3 = [arg_3_0,arg_3_1,]
arg_4 = "backward"
arg_5 = None
try:
  results["res_cpu"] = paddle.fft.fftn(arg_1,arg_2,arg_3,arg_4,arg_5,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_gpu"] = paddle.fft.fftn(arg_1,arg_2,arg_3,arg_4,arg_5,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
