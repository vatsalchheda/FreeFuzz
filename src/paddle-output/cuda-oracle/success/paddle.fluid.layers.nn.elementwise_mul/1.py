results = dict()
import paddle
arg_1_tensor = paddle.rand([4, 4, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([32], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = -20
try:
  results["res_cpu"] = paddle.fluid.layers.nn.elementwise_mul(arg_1,arg_2,axis=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.layers.nn.elementwise_mul(arg_1,arg_2,axis=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
