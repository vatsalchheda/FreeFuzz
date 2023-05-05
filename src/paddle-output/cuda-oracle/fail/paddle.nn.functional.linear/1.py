results = dict()
import paddle
arg_1_tensor = paddle.rand([2, 4, 128], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([128, 128], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([128], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = None
try:
  results["res_cpu"] = paddle.nn.functional.linear(x=arg_1,weight=arg_2,bias=arg_3,name=arg_4,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.linear(x=arg_1,weight=arg_2,bias=arg_3,name=arg_4,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
