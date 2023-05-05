results = dict()
import paddle
arg_1_tensor = paddle.randint(-4, 8, [3, 224, 224], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
try:
  results["res_cpu"] = paddle.nn.functional.one_hot(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.one_hot(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
