results = dict()
import paddle
arg_1_tensor = paddle.rand([1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 1
arg_2 = [arg_2_0,]
arg_3 = "paddleVarType"
arg_4 = 0.0
try:
  results["res_cpu"] = paddle.fluid.layers.tensor.fill_constant_batch_size_like(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,]
try:
  results["res_gpu"] = paddle.fluid.layers.tensor.fill_constant_batch_size_like(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
