results = dict()
import paddle
arg_1_tensor = paddle.rand([3, 3, 112, 112], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 63.0
arg_3_tensor = paddle.rand([192], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
try:
  results["res_cpu"] = paddle.scale(arg_1,scale=arg_2,bias=arg_3,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.scale(arg_1,scale=arg_2,bias=arg_3,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
