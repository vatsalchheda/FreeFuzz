results = dict()
import paddle
arg_1_tensor = paddle.randint(-16, 4096, [2, 3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1, 8192, [1, 1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3 = 111
arg_4 = -24
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
