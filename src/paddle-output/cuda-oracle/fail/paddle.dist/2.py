results = dict()
import paddle
arg_1_tensor = paddle.randint(-8192,8,[2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8192,512,[2, 2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = -inf
try:
  results["res_cpu"] = paddle.dist(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.dist(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
