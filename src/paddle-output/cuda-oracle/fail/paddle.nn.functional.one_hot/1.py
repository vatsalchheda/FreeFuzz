results = dict()
import paddle
arg_1_tensor = paddle.randint(-16384, 2048, [10, 2, 3], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2 = -1
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
