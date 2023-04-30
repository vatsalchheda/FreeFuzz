results = dict()
import paddle
arg_1_tensor = paddle.randint(-64,16384,[2, 3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4096,256,[3, 9, 10], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_0 = 3
arg_3_1 = 5
arg_3_2 = 9
arg_3_3 = 10
arg_3 = [arg_3_0,arg_3_1,arg_3_2,arg_3_3,]
try:
  results["res_cpu"] = paddle.scatter_nd(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = [arg_3_0,arg_3_1,arg_3_2,arg_3_3,]
try:
  results["res_gpu"] = paddle.scatter_nd(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
