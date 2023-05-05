results = dict()
import paddle
arg_1_tensor = paddle.rand([1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0_tensor = paddle.rand([100, 256, 1], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2_1_tensor = paddle.rand([3350, 1], dtype=paddle.float32)
arg_2_1 = arg_2_1_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = "builtinsset"
try:
  results["res_cpu"] = paddle.static.append_backward(loss=arg_1,parameter_list=arg_2,no_grad_set=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2_0 = arg_2_0_tensor.clone().cuda()
arg_2_1 = arg_2_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,]
try:
  results["res_gpu"] = paddle.static.append_backward(loss=arg_1,parameter_list=arg_2,no_grad_set=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
