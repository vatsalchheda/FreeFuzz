results = dict()
import paddle
arg_1_tensor = paddle.rand([10, 4, 8], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -16
try:
  results["res_cpu"] = paddle.fluid.layers.nn.topk(arg_1,k=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.layers.nn.topk(arg_1,k=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
