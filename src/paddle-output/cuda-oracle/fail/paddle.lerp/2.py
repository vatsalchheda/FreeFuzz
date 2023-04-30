results = dict()
import paddle
arg_1_tensor = paddle.randint(-8,16,[4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8,2,[4], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = 0.5
try:
  results["res_cpu"] = paddle.lerp(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.lerp(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
