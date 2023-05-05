results = dict()
import paddle
arg_1_tensor = paddle.rand([-1, 2048], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
arg_3 = 256
arg_4 = False
arg_5 = -10.0
arg_6 = "tanh"
arg_7 = "tanh"
try:
  results["res_cpu"] = paddle.fluid.layers.dynamic_lstmp(input=arg_1,size=arg_2,proj_size=arg_3,use_peepholes=arg_4,is_reverse=arg_5,cell_activation=arg_6,proj_activation=arg_7,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.layers.dynamic_lstmp(input=arg_1,size=arg_2,proj_size=arg_3,use_peepholes=arg_4,is_reverse=arg_5,cell_activation=arg_6,proj_activation=arg_7,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
