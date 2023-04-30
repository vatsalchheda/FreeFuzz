results = dict()
import paddle
arg_1_tensor = paddle.randint(-16384,256,[1, 2, 3, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 8
try:
  results["res_cpu"] = paddle.nn.functional.maxout(arg_1,groups=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.maxout(arg_1,groups=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
