results = dict()
import paddle
arg_1 = "tanh"
arg_2_tensor = paddle.rand([1, 200], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([1, 200], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = "tanh_grad"
arg_5_tensor = paddle.rand([1, 200], dtype=paddle.float32)
arg_5 = arg_5_tensor.clone()
try:
  results["res_cpu"] = paddle.static.py_func(func=arg_1,x=arg_2,out=arg_3,backward_func=arg_4,skip_vars_in_backward_input=arg_5,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
arg_5 = arg_5_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.static.py_func(func=arg_1,x=arg_2,out=arg_3,backward_func=arg_4,skip_vars_in_backward_input=arg_5,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
