results = dict()
import paddle
arg_1_tensor = paddle.randint(-1024,1024,[2, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 2
arg_2_1 = 0
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_cpu"] = paddle.fft.rfftn(arg_1,axes=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_gpu"] = paddle.fft.rfftn(arg_1,axes=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
