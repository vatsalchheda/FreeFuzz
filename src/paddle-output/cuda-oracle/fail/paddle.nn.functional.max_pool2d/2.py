results = dict()
import paddle
arg_1_tensor = paddle.randint(-16,32768,[-1, 20, 24, 24], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -18
arg_3 = 2
try:
  results["res_cpu"] = paddle.nn.functional.max_pool2d(arg_1,kernel_size=arg_2,stride=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.max_pool2d(arg_1,kernel_size=arg_2,stride=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
