results = dict()
import paddle
arg_1_tensor = paddle.rand([1, 3, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 64
arg_3 = -11.0
arg_4 = 0
arg_5 = True
arg_6 = False
arg_7 = None
try:
  results["res_cpu"] = paddle.nn.functional.max_pool1d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.max_pool1d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
