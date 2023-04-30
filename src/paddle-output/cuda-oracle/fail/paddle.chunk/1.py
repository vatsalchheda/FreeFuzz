results = dict()
import paddle
arg_1_tensor = paddle.randint(-1,8,[3, 9, 5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 45
arg_3 = -2
try:
  results["res_cpu"] = paddle.chunk(arg_1,chunks=arg_2,axis=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.chunk(arg_1,chunks=arg_2,axis=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
