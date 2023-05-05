results = dict()
import paddle
arg_1_tensor = paddle.rand([2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -1
arg_3 = False
try:
  results["res_cpu"] = paddle.fluid.layers.nn.reduce_max(arg_1,dim=arg_2,keep_dim=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.layers.nn.reduce_max(arg_1,dim=arg_2,keep_dim=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
