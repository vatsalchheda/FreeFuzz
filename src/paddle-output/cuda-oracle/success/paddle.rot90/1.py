results = dict()
import paddle
arg_1_tensor = paddle.randint(-256, 16, [2, 2, 2], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = "max"
arg_3_0 = -63
arg_3_1 = -16
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_cpu"] = paddle.rot90(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_3 = [arg_3_0,arg_3_1,]
try:
  results["res_gpu"] = paddle.rot90(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
