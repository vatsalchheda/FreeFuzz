results = dict()
import paddle
arg_1_tensor = paddle.randint(-512,8,[3, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8192,8,[4], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-2,64,[9, 2], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = True
try:
  results["res_cpu"] = paddle.scatter(arg_1,arg_2,arg_3,overwrite=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.scatter(arg_1,arg_2,arg_3,overwrite=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
