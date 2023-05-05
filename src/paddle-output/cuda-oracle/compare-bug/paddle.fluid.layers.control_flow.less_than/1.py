results = dict()
import paddle
arg_1_tensor = paddle.randint(-256, 4096, [1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4, 32768, [1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3 = False
arg_4_tensor = paddle.randint(0,2,[1])
arg_4 = arg_4_tensor.clone()
try:
  results["res_cpu"] = paddle.fluid.layers.control_flow.less_than(x=arg_1,y=arg_2,force_cpu=arg_3,cond=arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_4 = arg_4_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.layers.control_flow.less_than(x=arg_1,y=arg_2,force_cpu=arg_3,cond=arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
