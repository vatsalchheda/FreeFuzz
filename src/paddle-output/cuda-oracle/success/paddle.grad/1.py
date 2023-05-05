results = dict()
import paddle
arg_1_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0_tensor = paddle.rand([2], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
arg_3_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = True
arg_5 = True
try:
  results["res_cpu"] = paddle.grad(arg_1,arg_2,arg_3,create_graph=arg_4,allow_unused=arg_5,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2_0 = arg_2_0_tensor.clone().cuda()
arg_2 = [arg_2_0,]
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.grad(arg_1,arg_2,arg_3,create_graph=arg_4,allow_unused=arg_5,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
