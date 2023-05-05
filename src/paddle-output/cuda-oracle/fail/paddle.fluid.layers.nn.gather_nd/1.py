results = dict()
import paddle
arg_1_tensor = paddle.randint(-8,16,[4, 4], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32,8192,[4, 4, 2], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
try:
  results["res_cpu"] = paddle.fluid.layers.nn.gather_nd(arg_1,arg_2,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.layers.nn.gather_nd(arg_1,arg_2,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
