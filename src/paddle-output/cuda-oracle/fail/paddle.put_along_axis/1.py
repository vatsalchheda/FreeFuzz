results = dict()
import paddle
arg_1_tensor = paddle.randint(-4, 1, [2, 3], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4, 2048, [1, 1], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
arg_3 = 99
arg_4 = 0
try:
  results["res_cpu"] = paddle.put_along_axis(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.put_along_axis(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
