results = dict()
import paddle
arg_1_tensor = paddle.randint(-8,2048,[38], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = "sum"
try:
  results["res_cpu"] = paddle.fluid.layers.tensor.cast(x=arg_1,dtype=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.layers.tensor.cast(x=arg_1,dtype=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
