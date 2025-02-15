results = dict()
import paddle
arg_1_tensor = paddle.rand([1, 3, 32, 32, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 5
arg_3 = -2.0
arg_4 = 28
arg_5 = "max"
try:
  results["res_cpu"] = paddle.nn.functional.max_pool3d(arg_1,kernel_size=arg_2,stride=arg_3,padding=arg_4,return_mask=arg_5,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.max_pool3d(arg_1,kernel_size=arg_2,stride=arg_3,padding=arg_4,return_mask=arg_5,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
