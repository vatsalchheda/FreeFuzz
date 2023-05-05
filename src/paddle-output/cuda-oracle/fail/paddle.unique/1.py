results = dict()
import paddle
arg_1_tensor = paddle.randint(-128, 4, [6], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2 = 74
arg_3 = True
arg_4 = True
try:
  results["res_cpu"] = paddle.unique(arg_1,return_index=arg_2,return_inverse=arg_3,return_counts=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.unique(arg_1,return_index=arg_2,return_inverse=arg_3,return_counts=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
