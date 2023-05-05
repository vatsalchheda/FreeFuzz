results = dict()
import paddle
arg_1 = -17
arg_2_tensor = paddle.randint(-64, 4096, [1], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
arg_3 = 1
arg_4 = "paddleVarType"
try:
  results["res_cpu"] = paddle.fluid.layers.tensor.range(arg_1,arg_2,arg_3,dtype=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.layers.tensor.range(arg_1,arg_2,arg_3,dtype=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
