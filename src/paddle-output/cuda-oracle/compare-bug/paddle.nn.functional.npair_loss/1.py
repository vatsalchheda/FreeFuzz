results = dict()
import paddle
arg_1_tensor = paddle.rand([18, 6], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([18, 6], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([18], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = 0.002
try:
  results["res_cpu"] = paddle.nn.functional.npair_loss(arg_1,arg_2,arg_3,l2_reg=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.nn.functional.npair_loss(arg_1,arg_2,arg_3,l2_reg=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
