results = dict()
import paddle
arg_1_0_tensor = paddle.rand([-1, 28, 28], dtype=paddle.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
arg_2_0_tensor = paddle.rand([-1, 10], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
try:
  results["res_cpu"] = paddle.static.serialize_program(arg_1,arg_2,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1_0 = arg_1_0_tensor.clone().cuda()
arg_1 = [arg_1_0,]
arg_2_0 = arg_2_0_tensor.clone().cuda()
arg_2 = [arg_2_0,]
try:
  results["res_gpu"] = paddle.static.serialize_program(arg_1,arg_2,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
