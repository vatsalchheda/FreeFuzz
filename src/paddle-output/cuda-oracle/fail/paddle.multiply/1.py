results = dict()
import paddle
arg_1_tensor = paddle.rand([38, 512, 123], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(0,2,[2, 2])
arg_2 = arg_2_tensor.clone()
arg_3 = None
try:
  results["res_cpu"] = paddle.multiply(arg_1,arg_2,name=arg_3,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.multiply(arg_1,arg_2,name=arg_3,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
