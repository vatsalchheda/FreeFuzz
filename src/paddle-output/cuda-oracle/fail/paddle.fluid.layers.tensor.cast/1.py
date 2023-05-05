results = dict()
import paddle
arg_1_tensor = paddle.randint(-16384, 4096, [1], dtype=paddle.int32arg_1 = arg_1_tensor.clone()
arg_2 = "paddleVarType"
try:
  results["res_cpu"] = paddle.fluid.layers.tensor.cast(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.layers.tensor.cast(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
