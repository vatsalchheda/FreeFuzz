results = dict()
import paddle
arg_1_tensor = paddle.rand([1, 164, 1024], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 164
arg_2_1 = 1024
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_cpu"] = paddle.nn.functional.layer_norm(arg_1,arg_2,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_gpu"] = paddle.nn.functional.layer_norm(arg_1,arg_2,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
