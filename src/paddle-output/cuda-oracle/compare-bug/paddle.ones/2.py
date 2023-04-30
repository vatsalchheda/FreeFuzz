results = dict()
import paddle
arg_1_tensor = paddle.randint(-16,64,[2], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2 = "int32"
try:
  results["res_cpu"] = paddle.ones(shape=arg_1,dtype=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.ones(shape=arg_1,dtype=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
