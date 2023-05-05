results = dict()
import paddle
arg_1_tensor = paddle.rand([-1, 64], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([-1, 512], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([-1, 512], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
try:
  results["res_cpu"] = paddle.fluid.layers.lstm_unit(x_t=arg_1,hidden_t_prev=arg_2,cell_t_prev=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.layers.lstm_unit(x_t=arg_1,hidden_t_prev=arg_2,cell_t_prev=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
