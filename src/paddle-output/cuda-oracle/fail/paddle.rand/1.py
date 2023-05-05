results = dict()
import paddle
arg_1_0_tensor = paddle.randint(-16,256,[1], dtype=paddle.int32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = paddle.randint(-32,1,[1], dtype=paddle.int32)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
try:
  results["res_cpu"] = paddle.rand(shape=arg_1,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1_0 = arg_1_0_tensor.clone().cuda()
arg_1_1 = arg_1_1_tensor.clone().cuda()
arg_1 = [arg_1_0,arg_1_1,]
try:
  results["res_gpu"] = paddle.rand(shape=arg_1,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
