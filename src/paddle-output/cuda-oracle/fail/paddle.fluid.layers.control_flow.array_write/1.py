results = dict()
import paddle
arg_1_tensor = paddle.rand([-1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32768, 128, [1], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([-1, 128], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
try:
  results["res_cpu"] = paddle.fluid.layers.control_flow.array_write(arg_1,i=arg_2,array=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.layers.control_flow.array_write(arg_1,i=arg_2,array=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
