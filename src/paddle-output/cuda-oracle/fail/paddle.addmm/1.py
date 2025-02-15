results = dict()
import paddle
arg_1_tensor = paddle.rand([-1, 20, 12, 12], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([2, 2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = 0.5
arg_5 = 5.0
try:
  results["res_cpu"] = paddle.addmm(input=arg_1,x=arg_2,y=arg_3,beta=arg_4,alpha=arg_5,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.addmm(input=arg_1,x=arg_2,y=arg_3,beta=arg_4,alpha=arg_5,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
