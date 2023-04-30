results = dict()
import paddle
arg_1_tensor = paddle.randint(-256,1024,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1,256,[2, 1], dtype=paddle.float64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-8,32768,[2, 1], dtype=paddle.float64)
arg_3 = arg_3_tensor.clone()
try:
  results["res_cpu"] = paddle.lerp(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.lerp(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
