results = dict()
import paddle
arg_1_tensor = paddle.randint(-32, 1024, [1, 1], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4096, 16384, [60000], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-32768, 4096, [60000], dtype=paddle.int64arg_3 = arg_3_tensor.clone()
try:
  results["res_cpu"] = paddle.scatter(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.scatter(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
