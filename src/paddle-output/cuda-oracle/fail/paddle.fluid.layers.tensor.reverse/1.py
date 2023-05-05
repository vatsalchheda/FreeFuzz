results = dict()
import paddle
arg_1_tensor = paddle.rand([4, 23, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -19
try:
  results["res_cpu"] = paddle.fluid.layers.tensor.reverse(arg_1,axis=arg_2,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.layers.tensor.reverse(arg_1,axis=arg_2,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
