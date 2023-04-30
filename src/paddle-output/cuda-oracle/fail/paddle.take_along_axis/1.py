results = dict()
import paddle
arg_1_tensor = paddle.randint(-4096,256,[4, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4,512,[4, 1], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
arg_3 = 34
try:
  results["res_cpu"] = paddle.take_along_axis(arg_1,arg_2,axis=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.take_along_axis(arg_1,arg_2,axis=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
