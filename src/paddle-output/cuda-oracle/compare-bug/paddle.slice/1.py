results = dict()
import paddle
arg_1_tensor = paddle.randint(-4, 512, [64, 10], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 36.0
arg_2 = [arg_2_0,]
arg_3_0 = 0
arg_3 = [arg_3_0,]
arg_4_0_tensor = paddle.randint(-128, 32, [1], dtype=paddle.int32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
try:
  results["res_cpu"] = paddle.slice(arg_1,axes=arg_2,starts=arg_3,ends=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,]
arg_3 = [arg_3_0,]
arg_4_0 = arg_4_0_tensor.clone().cuda()
arg_4 = [arg_4_0,]
try:
  results["res_gpu"] = paddle.slice(arg_1,axes=arg_2,starts=arg_3,ends=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
